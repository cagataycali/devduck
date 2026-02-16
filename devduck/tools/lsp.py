"""LSP (Language Server Protocol) integration tool for DevDuck.

Provides real-time code intelligence by communicating with language servers:
- Diagnostics (type errors, missing imports, warnings)
- Code navigation (go-to-definition, find-references)
- Hover documentation
- Document symbols

Supports pyright, typescript-language-server, gopls, rust-analyzer, clangd
out of the box, with custom server configuration via DEVDUCK_LSP_SERVERS env var.

Two integration modes:
1. Explicit tool calls: lsp(action="diagnostics", file_path="...")
2. Auto-diagnostics hook: LSPDiagnosticsHook appends diagnostics to file-modifying
   tool results (opt-in via DEVDUCK_LSP_AUTO_DIAGNOSTICS=true)
"""

import atexit
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote as url_quote
from urllib.request import pathname2url

from strands import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global LSP server registry
# ---------------------------------------------------------------------------
# Keyed by language ID (e.g. "python", "typescript").
# Each entry:
#   process: subprocess.Popen
#   running: bool
#   capabilities: dict (from initialize response)
#   diagnostics: dict[str, list]  (file URI -> list of diagnostic dicts)
#   request_id: int
#   pending: dict[int, threading.Event]
#   responses: dict[int, dict]
#   lock: threading.Lock
#   reader_thread: threading.Thread
#   open_documents: set[str]  (file URIs currently open)
#   root_uri: str
LSP_SERVERS: dict[str, dict[str, Any]] = {}

# Default server configurations: language -> (command, args)
DEFAULT_SERVERS: dict[str, dict[str, Any]] = {
    "python": {"command": "pyright-langserver", "args": ["--stdio"]},
    "typescript": {"command": "typescript-language-server", "args": ["--stdio"]},
    "javascript": {"command": "typescript-language-server", "args": ["--stdio"]},
    "go": {"command": "gopls", "args": ["serve"]},
    "rust": {"command": "rust-analyzer", "args": []},
    "c": {"command": "clangd", "args": []},
    "cpp": {"command": "clangd", "args": []},
}

# File extension -> language ID mapping
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _path_to_uri(file_path: str) -> str:
    """Convert a file path to a file:// URI."""
    p = os.path.abspath(file_path)
    return "file://" + pathname2url(p)


def _uri_to_path(uri: str) -> str:
    """Convert a file:// URI back to a path."""
    if uri.startswith("file://"):
        from urllib.parse import unquote, urlparse

        parsed = urlparse(uri)
        return unquote(parsed.path)
    return uri


def _detect_language(file_path: str) -> Optional[str]:
    """Detect language ID from file extension."""
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_LANG.get(ext)


def _get_server_config(language: str) -> Optional[dict]:
    """Get server config, checking custom config first, then defaults."""
    # Check custom config from env var
    custom = os.getenv("DEVDUCK_LSP_SERVERS", "")
    if custom:
        try:
            custom_servers = json.loads(custom)
            if language in custom_servers:
                return custom_servers[language]
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in DEVDUCK_LSP_SERVERS env var")

    return DEFAULT_SERVERS.get(language)


def _format_diagnostic(diag: dict) -> str:
    """Format a single diagnostic for display."""
    rng = diag.get("range", {})
    start = rng.get("start", {})
    line = start.get("line", 0) + 1  # Convert back to 1-based
    char = start.get("character", 0) + 1
    severity_map = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}
    sev = severity_map.get(diag.get("severity", 1), "ERROR")
    source = diag.get("source", "")
    msg = diag.get("message", "")
    prefix = f"[{source}] " if source else ""
    return f"  L{line}:{char} {sev}: {prefix}{msg}"


# ---------------------------------------------------------------------------
# JSON-RPC transport
# ---------------------------------------------------------------------------


def _encode_message(msg: dict) -> bytes:
    """Encode a JSON-RPC message with Content-Length header."""
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _send_message(process: subprocess.Popen, msg: dict) -> None:
    """Send a JSON-RPC message to the LSP server's stdin."""
    data = _encode_message(msg)
    process.stdin.write(data)
    process.stdin.flush()


def _read_message(stdout) -> Optional[dict]:
    """Read a single JSON-RPC message from the stream.

    Returns None on EOF or parse error.
    """
    headers = {}
    while True:
        line = stdout.readline()
        if not line:
            return None  # EOF
        line = line.decode("ascii", errors="replace").strip()
        if not line:
            break  # End of headers
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

    content_length = int(headers.get("Content-Length", 0))
    if content_length == 0:
        return None

    body = b""
    while len(body) < content_length:
        chunk = stdout.read(content_length - len(body))
        if not chunk:
            return None  # EOF
        body += chunk

    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        logger.warning("Failed to parse LSP message body")
        return None


# ---------------------------------------------------------------------------
# Background reader thread
# ---------------------------------------------------------------------------


def _reader_loop(language: str) -> None:
    """Background thread that reads LSP server stdout.

    Dispatches responses to pending requests and stores diagnostics notifications.
    """
    server = LSP_SERVERS.get(language)
    if not server:
        return

    process = server["process"]
    while server.get("running", False):
        try:
            msg = _read_message(process.stdout)
            if msg is None:
                # EOF — server died
                logger.info(f"LSP reader for {language}: EOF, server stopped")
                server["running"] = False
                break

            msg_id = msg.get("id")
            method = msg.get("method", "")

            if msg_id is not None and msg_id in server["pending"]:
                # Response to a request we sent
                with server["lock"]:
                    server["responses"][msg_id] = msg
                    event = server["pending"].pop(msg_id, None)
                if event:
                    event.set()

            elif method == "textDocument/publishDiagnostics":
                # Diagnostics notification
                params = msg.get("params", {})
                uri = params.get("uri", "")
                diags = params.get("diagnostics", [])
                with server["lock"]:
                    server["diagnostics"][uri] = diags

            elif method == "window/logMessage" or method == "window/showMessage":
                # Log server messages
                params = msg.get("params", {})
                log_msg = params.get("message", "")
                logger.debug(f"LSP [{language}]: {log_msg}")

            elif msg_id is not None and method:
                # Server-initiated request (has both id and method).
                # We must respond or the server blocks waiting.
                logger.debug(f"LSP [{language}]: server request {method}, responding null")
                try:
                    response = {"jsonrpc": "2.0", "id": msg_id, "result": None}
                    _send_message(process, response)
                except Exception:
                    pass

            elif msg_id is not None and "method" not in msg:
                # Response to a request we may have already timed out on
                with server["lock"]:
                    server["responses"][msg_id] = msg

        except Exception as e:
            if server.get("running", False):
                logger.warning(f"LSP reader error for {language}: {e}")
            break


# ---------------------------------------------------------------------------
# Request / notification helpers
# ---------------------------------------------------------------------------


def _next_id(language: str) -> int:
    """Get the next request ID for a language server."""
    server = LSP_SERVERS[language]
    with server["lock"]:
        server["request_id"] += 1
        return server["request_id"]


def _send_request(language: str, method: str, params: dict, timeout: float = 30.0) -> dict:
    """Send a JSON-RPC request and wait for the response."""
    server = LSP_SERVERS[language]
    req_id = _next_id(language)
    event = threading.Event()

    with server["lock"]:
        server["pending"][req_id] = event

    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    _send_message(server["process"], msg)

    if not event.wait(timeout=timeout):
        with server["lock"]:
            server["pending"].pop(req_id, None)
        raise TimeoutError(f"LSP request {method} timed out after {timeout}s")

    with server["lock"]:
        response = server["responses"].pop(req_id, {})

    if "error" in response:
        err = response["error"]
        raise RuntimeError(f"LSP error ({err.get('code')}): {err.get('message')}")

    return response.get("result", {})


def _send_notification(language: str, method: str, params: dict) -> None:
    """Send a JSON-RPC notification (no response expected)."""
    server = LSP_SERVERS[language]
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    _send_message(server["process"], msg)


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------


def _read_file_content(file_path: str) -> str:
    """Read file content, returning empty string on error."""
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _ensure_document_open(language: str, file_path: str) -> str:
    """Open a document in the LSP server if not already open. Returns the URI."""
    server = LSP_SERVERS[language]
    uri = _path_to_uri(file_path)

    if uri not in server["open_documents"]:
        lang_id = language
        # typescript-language-server expects "typescriptreact" for .tsx etc.
        ext = Path(file_path).suffix.lower()
        if ext == ".tsx":
            lang_id = "typescriptreact"
        elif ext == ".jsx":
            lang_id = "javascriptreact"

        content = _read_file_content(file_path)
        _send_notification(language, "textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": lang_id,
                "version": 1,
                "text": content,
            }
        })
        with server["lock"]:
            server["open_documents"].add(uri)

    return uri


def _refresh_document(language: str, file_path: str) -> str:
    """Close and re-open a document so the server picks up on-disk changes."""
    server = LSP_SERVERS[language]
    uri = _path_to_uri(file_path)

    # Close if open
    if uri in server["open_documents"]:
        _send_notification(language, "textDocument/didClose", {
            "textDocument": {"uri": uri}
        })
        with server["lock"]:
            server["open_documents"].discard(uri)

    # Re-open with fresh content
    return _ensure_document_open(language, file_path)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _start_server(language: str, root_path: Optional[str] = None) -> dict:
    """Start an LSP server for the given language."""
    if language in LSP_SERVERS and LSP_SERVERS[language].get("running"):
        return {
            "status": "success",
            "content": [{"text": f"LSP server for {language} is already running."}],
        }

    config = _get_server_config(language)
    if not config:
        return {
            "status": "error",
            "content": [{"text": f"No LSP server configured for language: {language}. "
                         f"Supported: {', '.join(DEFAULT_SERVERS.keys())}. "
                         f"Or set DEVDUCK_LSP_SERVERS env var with custom config."}],
        }

    cmd = config["command"]
    if not shutil.which(cmd):
        install_hints = {
            "pyright-langserver": "pip install pyright",
            "typescript-language-server": "npm install -g typescript-language-server typescript",
            "gopls": "go install golang.org/x/tools/gopls@latest",
            "rust-analyzer": "rustup component add rust-analyzer",
            "clangd": "apt install clangd  (or brew install llvm)",
        }
        hint = install_hints.get(cmd, f"Install {cmd} and ensure it's on PATH")
        return {
            "status": "error",
            "content": [{"text": f"LSP server binary not found: {cmd}\nInstall: {hint}"}],
        }

    root = root_path or os.getcwd()
    root_uri = _path_to_uri(root)

    try:
        process = subprocess.Popen(
            [cmd] + config.get("args", []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=root,
        )
    except OSError as e:
        return {
            "status": "error",
            "content": [{"text": f"Failed to start LSP server: {e}"}],
        }

    server_entry = {
        "process": process,
        "running": True,
        "capabilities": {},
        "diagnostics": {},
        "request_id": 0,
        "pending": {},
        "responses": {},
        "lock": threading.Lock(),
        "reader_thread": None,
        "open_documents": set(),
        "root_uri": root_uri,
        "language": language,
        "command": cmd,
    }
    LSP_SERVERS[language] = server_entry

    # Start reader thread
    reader = threading.Thread(
        target=_reader_loop, args=(language,), daemon=True, name=f"lsp-reader-{language}"
    )
    server_entry["reader_thread"] = reader
    reader.start()

    # Send initialize request
    try:
        result = _send_request(language, "initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "rootPath": root,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {"relatedInformation": True},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": True},
                    "references": {},
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                },
            },
        })
        server_entry["capabilities"] = result.get("capabilities", {})

        # Send initialized notification
        _send_notification(language, "initialized", {})

    except Exception as e:
        # Cleanup on failure
        process.terminate()
        process.wait(timeout=5)
        server_entry["running"] = False
        LSP_SERVERS.pop(language, None)
        return {
            "status": "error",
            "content": [{"text": f"LSP initialization failed: {e}"}],
        }

    return {
        "status": "success",
        "content": [{"text": f"LSP server started for {language} (root: {root})\n"
                     f"Server: {cmd} {' '.join(config.get('args', []))}\n"
                     f"Ready for diagnostics, navigation, and hover."}],
    }


def _stop_server(language: str) -> dict:
    """Stop an LSP server for the given language."""
    if language not in LSP_SERVERS or not LSP_SERVERS[language].get("running"):
        return {
            "status": "error",
            "content": [{"text": f"No running LSP server for {language}."}],
        }

    server = LSP_SERVERS[language]
    server["running"] = False

    try:
        # Send shutdown request
        _send_request(language, "shutdown", {}, timeout=5.0)
        # Send exit notification
        _send_notification(language, "exit", {})
    except Exception:
        pass  # Best effort

    # Terminate process
    process = server["process"]
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass

    LSP_SERVERS.pop(language, None)
    return {
        "status": "success",
        "content": [{"text": f"LSP server for {language} stopped."}],
    }


def _shutdown_all() -> None:
    """Shutdown all running LSP servers. Called via atexit."""
    for language in list(LSP_SERVERS.keys()):
        try:
            _stop_server(language)
        except Exception:
            pass


atexit.register(_shutdown_all)


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def _action_diagnostics(file_path: str, language: Optional[str] = None) -> dict:
    """Get diagnostics for a file."""
    if not file_path:
        return {"status": "error", "content": [{"text": "file_path is required for diagnostics."}]}

    lang = language or _detect_language(file_path)
    if not lang:
        return {"status": "error", "content": [{"text": f"Cannot detect language for: {file_path}"}]}

    if lang not in LSP_SERVERS or not LSP_SERVERS[lang].get("running"):
        return {"status": "error", "content": [{"text": f"No running LSP server for {lang}. Use action='start' first."}]}

    uri = _ensure_document_open(lang, file_path)
    # Wait for server to process the file
    time.sleep(1.0)

    diags = LSP_SERVERS[lang]["diagnostics"].get(uri, [])
    if not diags:
        return {
            "status": "success",
            "content": [{"text": f"No diagnostics for {file_path}. File is clean."}],
        }

    errors = [d for d in diags if d.get("severity") == 1]
    warnings = [d for d in diags if d.get("severity") == 2]
    infos = [d for d in diags if d.get("severity") == 3]
    hints = [d for d in diags if d.get("severity") == 4]

    lines = [f"Diagnostics for {file_path}: {len(errors)} error(s), {len(warnings)} warning(s), "
             f"{len(infos)} info(s), {len(hints)} hint(s)"]

    for d in diags:
        lines.append(_format_diagnostic(d))

    return {
        "status": "success",
        "content": [{"text": "\n".join(lines)}],
    }


def _action_goto_definition(file_path: str, line: int, character: int,
                            language: Optional[str] = None) -> dict:
    """Go to definition at a position."""
    lang = language or _detect_language(file_path)
    if not lang or lang not in LSP_SERVERS or not LSP_SERVERS[lang].get("running"):
        return {"status": "error", "content": [{"text": f"No running LSP server. Start one first."}]}

    uri = _ensure_document_open(lang, file_path)

    try:
        result = _send_request(lang, "textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1},  # Convert to 0-based
        })
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Definition lookup failed: {e}"}]}

    if not result:
        return {"status": "success", "content": [{"text": "No definition found at that position."}]}

    # Result can be Location, Location[], or LocationLink[]
    locations = result if isinstance(result, list) else [result]
    lines = []
    for loc in locations:
        target_uri = loc.get("uri") or loc.get("targetUri", "")
        target_range = loc.get("range") or loc.get("targetRange", {})
        start = target_range.get("start", {})
        path = _uri_to_path(target_uri)
        l = start.get("line", 0) + 1
        c = start.get("character", 0) + 1
        lines.append(f"{path}:{l}:{c}")

    return {
        "status": "success",
        "content": [{"text": "Definition(s) found:\n" + "\n".join(lines)}],
    }


def _action_find_references(file_path: str, line: int, character: int,
                            language: Optional[str] = None) -> dict:
    """Find all references to the symbol at a position."""
    lang = language or _detect_language(file_path)
    if not lang or lang not in LSP_SERVERS or not LSP_SERVERS[lang].get("running"):
        return {"status": "error", "content": [{"text": "No running LSP server. Start one first."}]}

    uri = _ensure_document_open(lang, file_path)

    try:
        result = _send_request(lang, "textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1},
            "context": {"includeDeclaration": True},
        })
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Find references failed: {e}"}]}

    if not result:
        return {"status": "success", "content": [{"text": "No references found."}]}

    locations = result if isinstance(result, list) else [result]
    truncated = len(locations) > 50
    locations = locations[:50]

    lines = []
    for loc in locations:
        target_uri = loc.get("uri", "")
        rng = loc.get("range", {})
        start = rng.get("start", {})
        path = _uri_to_path(target_uri)
        l = start.get("line", 0) + 1
        c = start.get("character", 0) + 1
        lines.append(f"{path}:{l}:{c}")

    header = f"Found {len(lines)} reference(s)"
    if truncated:
        header += f" (showing first 50 of {len(result)})"
    header += ":"

    return {
        "status": "success",
        "content": [{"text": header + "\n" + "\n".join(lines)}],
    }


def _action_hover(file_path: str, line: int, character: int,
                  language: Optional[str] = None) -> dict:
    """Get hover information at a position."""
    lang = language or _detect_language(file_path)
    if not lang or lang not in LSP_SERVERS or not LSP_SERVERS[lang].get("running"):
        return {"status": "error", "content": [{"text": "No running LSP server. Start one first."}]}

    uri = _ensure_document_open(lang, file_path)

    try:
        result = _send_request(lang, "textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1},
        })
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Hover failed: {e}"}]}

    if not result:
        return {"status": "success", "content": [{"text": "No hover information at that position."}]}

    contents = result.get("contents", "")
    if isinstance(contents, dict):
        # MarkupContent: {"kind": "markdown", "value": "..."}
        text = contents.get("value", str(contents))
    elif isinstance(contents, list):
        # MarkedString[]
        parts = []
        for c in contents:
            if isinstance(c, dict):
                parts.append(c.get("value", str(c)))
            else:
                parts.append(str(c))
        text = "\n---\n".join(parts)
    else:
        text = str(contents)

    return {
        "status": "success",
        "content": [{"text": f"Hover info:\n{text}"}],
    }


def _action_document_symbols(file_path: str, language: Optional[str] = None) -> dict:
    """Get document symbols (outline) for a file."""
    lang = language or _detect_language(file_path)
    if not lang or lang not in LSP_SERVERS or not LSP_SERVERS[lang].get("running"):
        return {"status": "error", "content": [{"text": "No running LSP server. Start one first."}]}

    uri = _ensure_document_open(lang, file_path)

    try:
        result = _send_request(lang, "textDocument/documentSymbol", {
            "textDocument": {"uri": uri},
        })
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Document symbols failed: {e}"}]}

    if not result:
        return {"status": "success", "content": [{"text": "No symbols found."}]}

    symbols = result if isinstance(result, list) else [result]

    # Symbol kind mapping (LSP spec)
    kind_map = {
        1: "File", 2: "Module", 3: "Namespace", 4: "Package",
        5: "Class", 6: "Method", 7: "Property", 8: "Field",
        9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
        13: "Variable", 14: "Constant", 15: "String", 16: "Number",
        17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
        21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
        25: "Operator", 26: "TypeParameter",
    }

    def _format_symbol(sym: dict, indent: int = 0) -> list[str]:
        kind_num = sym.get("kind", 0)
        kind = kind_map.get(kind_num, f"Kind({kind_num})")
        name = sym.get("name", "?")
        rng = sym.get("range") or sym.get("location", {}).get("range", {})
        start = rng.get("start", {})
        line = start.get("line", 0) + 1
        prefix = "  " * indent
        lines = [f"{prefix}{kind} {name} (line {line})"]
        for child in sym.get("children", []):
            lines.extend(_format_symbol(child, indent + 1))
        return lines

    output_lines = [f"Symbols in {file_path}:"]
    for sym in symbols:
        output_lines.extend(_format_symbol(sym))

    return {
        "status": "success",
        "content": [{"text": "\n".join(output_lines)}],
    }


def _action_status() -> dict:
    """Get status of all LSP servers."""
    if not LSP_SERVERS:
        return {
            "status": "success",
            "content": [{"text": "No LSP servers running.\n"
                         f"Supported languages: {', '.join(DEFAULT_SERVERS.keys())}"}],
        }

    lines = ["LSP Server Status:"]
    for lang, server in LSP_SERVERS.items():
        running = server.get("running", False)
        cmd = server.get("command", "?")
        root = _uri_to_path(server.get("root_uri", ""))
        diag_count = sum(len(d) for d in server.get("diagnostics", {}).values())
        docs = len(server.get("open_documents", set()))
        status_str = "running" if running else "stopped"
        lines.append(f"  {lang}: {status_str} ({cmd})")
        lines.append(f"    Root: {root}")
        lines.append(f"    Open documents: {docs}, Total diagnostics: {diag_count}")

    return {
        "status": "success",
        "content": [{"text": "\n".join(lines)}],
    }


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


@tool
def lsp(
    action: str,
    file_path: str = "",
    language: str = "",
    line: int = 0,
    character: int = 0,
    root_path: str = "",
    command: str = "",
    args: str = "",
) -> dict:
    """Interact with Language Server Protocol servers for real-time code intelligence.

    Provides diagnostics (type errors, warnings), code navigation (go-to-definition,
    find-references), hover documentation, and document symbols.

    Args:
        action: Action to perform. One of:
            - "start": Start an LSP server for a language
            - "stop": Stop an LSP server
            - "diagnostics": Get diagnostics (errors/warnings) for a file
            - "goto_definition": Go to definition at a position
            - "find_references": Find all references to a symbol
            - "hover": Get hover/type info at a position
            - "document_symbols": Get symbols outline for a file
            - "status": Show status of all running LSP servers
        file_path: Path to the file (required for diagnostics, navigation, hover)
        language: Language ID (python, typescript, go, rust, c, cpp). Auto-detected from
            file extension if not provided.
        line: Line number (1-based) for navigation/hover actions
        character: Column number (1-based) for navigation/hover actions
        root_path: Project root directory (defaults to cwd). Used when starting a server.
        command: Custom server command (overrides default for the language)
        args: Custom server arguments as space-separated string

    Returns:
        dict with "status" and "content" keys
    """
    try:
        if action == "start":
            lang = language
            if not lang and file_path:
                lang = _detect_language(file_path)
            if not lang:
                return {
                    "status": "error",
                    "content": [{"text": "language is required for start action. "
                                 f"Supported: {', '.join(DEFAULT_SERVERS.keys())}"}],
                }
            # Allow custom command override
            if command:
                custom_args = args.split() if args else []
                DEFAULT_SERVERS[lang] = {"command": command, "args": custom_args}

            return _start_server(lang, root_path=root_path or None)

        elif action == "stop":
            lang = language
            if not lang and file_path:
                lang = _detect_language(file_path)
            if not lang:
                # Stop all
                if not LSP_SERVERS:
                    return {"status": "success", "content": [{"text": "No LSP servers running."}]}
                results = []
                for l in list(LSP_SERVERS.keys()):
                    r = _stop_server(l)
                    results.append(r["content"][0]["text"])
                return {"status": "success", "content": [{"text": "\n".join(results)}]}
            return _stop_server(lang)

        elif action == "diagnostics":
            return _action_diagnostics(file_path, language=language or None)

        elif action == "goto_definition":
            if not file_path or not line:
                return {"status": "error", "content": [{"text": "file_path and line are required for goto_definition."}]}
            return _action_goto_definition(file_path, line, character or 1, language=language or None)

        elif action == "find_references":
            if not file_path or not line:
                return {"status": "error", "content": [{"text": "file_path and line are required for find_references."}]}
            return _action_find_references(file_path, line, character or 1, language=language or None)

        elif action == "hover":
            if not file_path or not line:
                return {"status": "error", "content": [{"text": "file_path and line are required for hover."}]}
            return _action_hover(file_path, line, character or 1, language=language or None)

        elif action == "document_symbols":
            if not file_path:
                return {"status": "error", "content": [{"text": "file_path is required for document_symbols."}]}
            return _action_document_symbols(file_path, language=language or None)

        elif action == "status":
            return _action_status()

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}\n\n"
                             "Valid actions: start, stop, diagnostics, goto_definition, "
                             "find_references, hover, document_symbols, status"}],
            }

    except Exception as e:
        logger.exception(f"LSP tool error (action={action})")
        return {
            "status": "error",
            "content": [{"text": f"LSP error: {e}"}],
        }


# ---------------------------------------------------------------------------
# LSPDiagnosticsHook — auto-appends diagnostics to file-modifying tool results
# ---------------------------------------------------------------------------

try:
    from strands.hooks import HookProvider, HookRegistry
    from strands.hooks.events import AfterToolCallEvent

    _HAS_HOOKS = True
except ImportError:
    _HAS_HOOKS = False


if _HAS_HOOKS:
    class LSPDiagnosticsHook(HookProvider):
        """Auto-appends LSP diagnostics to file-modifying tool results.

        When registered, this hook fires after tools like shell, editor, write, and edit
        complete. If any LSP servers are running, it refreshes the modified files and
        appends any errors/warnings to the tool result so the agent can self-correct.

        Enable via: DEVDUCK_LSP_AUTO_DIAGNOSTICS=true
        """

        FILE_TOOLS = {"shell", "editor", "write", "edit", "file_write"}

        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(AfterToolCallEvent, self._on_tool_complete)

        def _on_tool_complete(self, event: AfterToolCallEvent) -> None:
            tool_name = event.tool_use.get("name", "").lower()
            if tool_name not in self.FILE_TOOLS:
                return
            if not LSP_SERVERS:
                return

            # Extract file paths from tool input
            file_paths = self._extract_file_paths(event.tool_use)

            # Collect diagnostics for modified files
            diag_summary: list[str] = []
            for fp in file_paths:
                lang = _detect_language(fp)
                if lang and lang in LSP_SERVERS and LSP_SERVERS[lang].get("running"):
                    _refresh_document(lang, fp)
                    time.sleep(0.5)
                    uri = _path_to_uri(fp)
                    diags = LSP_SERVERS[lang]["diagnostics"].get(uri, [])
                    errors = [d for d in diags if d.get("severity") == 1]
                    warnings = [d for d in diags if d.get("severity") == 2]
                    if errors or warnings:
                        diag_summary.append(
                            f"{fp}: {len(errors)} error(s), {len(warnings)} warning(s)"
                        )
                        for d in errors[:5]:  # Cap at 5 errors per file
                            diag_summary.append(_format_diagnostic(d))

            # Append to tool result
            if diag_summary:
                existing = event.result.get("content", [])
                existing.append({"text": "\n[LSP Diagnostics]\n" + "\n".join(diag_summary)})
                event.result["content"] = existing

        @staticmethod
        def _extract_file_paths(tool_use: dict) -> list[str]:
            """Extract file paths from a tool_use input dict."""
            paths: list[str] = []
            tool_input = tool_use.get("input", {})

            # Common parameter names for file paths across tools
            for key in ("file_path", "path", "filename", "file", "filepath"):
                val = tool_input.get(key, "")
                if val and isinstance(val, str) and os.path.isfile(val):
                    paths.append(val)

            # Shell command — try to extract paths from the command string
            cmd = tool_input.get("command", "")
            if cmd and isinstance(cmd, str):
                # Simple heuristic: look for tokens that are existing file paths
                for token in cmd.split():
                    token = token.strip("\"'")
                    if os.path.isfile(token):
                        lang = _detect_language(token)
                        if lang:
                            paths.append(token)

            return list(set(paths))  # Deduplicate

else:
    # Fallback when strands.hooks is not available
    class LSPDiagnosticsHook:  # type: ignore[no-redef]
        """Stub for environments without strands.hooks support."""

        def __init__(self) -> None:
            logger.warning(
                "LSPDiagnosticsHook requires strands.hooks (strands-agents >= 1.x). "
                "Auto-diagnostics will not be available."
            )

        def register_hooks(self, registry: Any) -> None:
            pass
