"""
🔌 OpenAPI Tool — Universal API client that reads any OpenAPI/Swagger spec.

Features:
- Auto-discovers endpoints from OpenAPI 3.x / Swagger 2.0 specs
- Supports both JSON and YAML spec formats (.json, .yaml, .yml)
- Loads from URLs, local file paths, and GitHub blob URLs (auto-converts to raw)
- Handles OAuth2 (authorization_code, client_credentials, implicit, password)
- Handles API Key (header, query, cookie)
- Handles Bearer token, Basic auth
- Persistent token storage with auto-refresh
- The LLM just says what it wants; this tool figures out the API call

Usage:
    openapi(action="load", spec_url="https://api.example.com/openapi.json")
    openapi(action="load", spec_url="https://api.example.com/openapi.yaml")
    openapi(action="load", spec_url="./local-spec.yml")
    openapi(action="list")  # show available operations
    openapi(action="call", operation="listUsers", params='{"limit": 10}')
    openapi(action="auth", provider="github", client_id="...", client_secret="...")
"""

import os
import json
import time
import hashlib
import threading
import webbrowser
import base64
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from strands import tool

# Token & spec storage
_STORE_DIR = Path.home() / ".devduck" / "openapi"
_STORE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory state
_LOADED_SPECS: Dict[str, dict] = {}  # alias -> parsed spec
_AUTH_TOKENS: Dict[str, dict] = {}   # provider -> {access_token, refresh_token, expires_at, type}
_BASE_URLS: Dict[str, str] = {}      # alias -> base url


def _spec_id(url_or_alias: str) -> str:
    """Generate a short stable ID for a spec."""
    return hashlib.md5(url_or_alias.encode()).hexdigest()[:8]


def _save_token(provider: str, token_data: dict):
    """Persist token to disk."""
    token_file = _STORE_DIR / f"token_{provider}.json"
    token_file.write_text(json.dumps(token_data, indent=2))
    token_file.chmod(0o600)
    _AUTH_TOKENS[provider] = token_data


def _load_token(provider: str) -> Optional[dict]:
    """Load token from disk."""
    if provider in _AUTH_TOKENS:
        return _AUTH_TOKENS[provider]
    token_file = _STORE_DIR / f"token_{provider}.json"
    if token_file.exists():
        data = json.loads(token_file.read_text())
        _AUTH_TOKENS[provider] = data
        return data
    return None


def _is_token_expired(token_data: dict) -> bool:
    """Check if token is expired (with 60s buffer)."""
    expires_at = token_data.get("expires_at", 0)
    if expires_at == 0:
        return False  # No expiry = assume valid
    return time.time() > (expires_at - 60)


def _parse_spec_content(raw: str, source: str = "") -> dict:
    """Parse raw spec content as JSON or YAML.

    Args:
        raw: Raw text content of the spec
        source: Source URL or path (used for format hinting)

    Returns:
        Parsed spec as dict
    """
    # Try JSON first (faster, more common in API responses)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try YAML (covers .yaml, .yml, and YAML-formatted specs)
    try:
        import yaml
        result = yaml.safe_load(raw)
        if isinstance(result, dict):
            return result
    except ImportError:
        # If PyYAML not installed and source looks like YAML, give helpful error
        if any(source.endswith(ext) for ext in (".yaml", ".yml")):
            raise RuntimeError(
                f"YAML spec detected but PyYAML not installed. "
                f"Install with: pip install pyyaml"
            )
    except Exception:
        pass

    raise RuntimeError(
        f"Could not parse spec as JSON or YAML from: {source or 'unknown source'}"
    )


def _fetch_spec(url: str) -> dict:
    """Fetch and parse an OpenAPI spec from URL or local file path.

    Supports:
    - JSON specs (.json or content-type application/json)
    - YAML specs (.yaml, .yml or content-type application/x-yaml)
    - Local file paths (absolute or relative)
    - HTTP/HTTPS URLs
    - GitHub raw URLs (auto-converts blob URLs)
    """
    # Handle local file paths
    if not url.startswith(("http://", "https://")):
        local_path = Path(url).expanduser().resolve()
        if local_path.exists():
            raw = local_path.read_text(encoding="utf-8")
            return _parse_spec_content(raw, str(local_path))
        raise FileNotFoundError(f"Local spec file not found: {url}")

    # Auto-convert GitHub blob URLs to raw URLs
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Determine Accept header based on URL extension
    accept_header = "application/json, application/x-yaml, text/yaml, text/plain"

    try:
        req = urllib.request.Request(url, headers={
            "Accept": accept_header,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) DevDuck/1.0",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        return _parse_spec_content(raw, url)
    except Exception as first_err:
        # Fallback to curl (bypasses Cloudflare bot detection)
        import subprocess
        try:
            result = subprocess.run(
                ["curl", "-sL", "-H", f"Accept: {accept_header}", url],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr}")
            return _parse_spec_content(result.stdout, url)
        except Exception as curl_err:
            raise RuntimeError(f"Failed to fetch spec from {url}: {first_err}; curl fallback: {curl_err}")


def _resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve a $ref pointer in the spec."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for part in parts:
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[part]
    return node


def _get_base_url(spec: dict, spec_url: str = "") -> str:
    """Extract base URL from spec."""
    # OpenAPI 3.x
    servers = spec.get("servers", [])
    if servers:
        server_url = servers[0].get("url", "").rstrip("/")
        # If server URL is relative, combine with spec URL origin
        if server_url and not server_url.startswith("http"):
            if spec_url:
                parsed = urllib.parse.urlparse(spec_url)
                server_url = f"{parsed.scheme}://{parsed.netloc}{server_url}"
        return server_url
    # Swagger 2.0
    host = spec.get("host", "")
    basepath = spec.get("basePath", "")
    schemes = spec.get("schemes", ["https"])
    if host:
        return f"{schemes[0]}://{host}{basepath}".rstrip("/")
    # Fallback: derive from spec URL
    if spec_url:
        parsed = urllib.parse.urlparse(spec_url)
        return f"{parsed.scheme}://{parsed.netloc}"
    return ""


def _extract_operations(spec: dict) -> Dict[str, dict]:
    """Extract all operations from the spec into a flat dict."""
    operations = {}
    paths = spec.get("paths", {})
    for path, methods in paths.items():
        # Handle path-level parameters
        path_params = methods.get("parameters", [])
        for method_name, method_detail in methods.items():
            if method_name not in ("get", "post", "put", "delete", "patch", "head", "options"):
                continue
            if not isinstance(method_detail, dict):
                continue

            op_id = method_detail.get("operationId", f"{method_name}_{path.replace('/', '_').strip('_')}")

            # Merge path-level and operation-level parameters
            op_params = list(path_params) + method_detail.get("parameters", [])
            # Resolve $ref params
            resolved_params = []
            for p in op_params:
                if "$ref" in p:
                    try:
                        p = _resolve_ref(spec, p["$ref"])
                    except Exception:
                        pass
                resolved_params.append(p)

            operations[op_id] = {
                "method": method_name.upper(),
                "path": path,
                "summary": method_detail.get("summary", ""),
                "description": method_detail.get("description", ""),
                "parameters": resolved_params,
                "requestBody": method_detail.get("requestBody", {}),
                "security": method_detail.get("security", spec.get("security", [])),
                "tags": method_detail.get("tags", []),
            }
    return operations


def _extract_security_schemes(spec: dict) -> dict:
    """Extract security schemes from spec."""
    # OpenAPI 3.x
    components = spec.get("components", {})
    schemes = components.get("securitySchemes", {})
    if schemes:
        return schemes
    # Swagger 2.0
    return spec.get("securityDefinitions", {})


def _do_oauth2_authorization_code(
    scheme: dict,
    client_id: str,
    client_secret: str,
    scopes: list = None,
    redirect_uri: str = None,
    redirect_port: int = None,
) -> dict:
    """Perform OAuth2 authorization code flow with local callback server.

    Supports custom redirect_uri (e.g. from env vars like SPOTIFY_REDIRECT_URI).
    Auto-detects port from redirect_uri if not explicitly provided.
    """
    flows = scheme.get("flows", {})
    auth_code_flow = flows.get("authorizationCode", {})
    auth_url = auth_code_flow.get("authorizationUrl", "")
    token_url = auth_code_flow.get("tokenUrl", "")
    available_scopes = auth_code_flow.get("scopes", {})

    if not auth_url or not token_url:
        return {"error": "Missing authorizationUrl or tokenUrl in spec"}

    if scopes is None:
        scopes = list(available_scopes.keys())

    # Determine redirect URI and port
    if redirect_uri is None:
        port = redirect_port or 9876
        redirect_uri = f"http://127.0.0.1:{port}/callback"
    else:
        # Parse port from provided redirect_uri
        parsed_ruri = urllib.parse.urlparse(redirect_uri)
        port = parsed_ruri.port or redirect_port or 9876

    state = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    # Build authorization URL
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
        "show_dialog": "true",
    }
    full_auth_url = f"{auth_url}?{urllib.parse.urlencode(params)}"

    # Start local callback server
    auth_code = {"code": None, "error": None}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            if qs.get("state", [None])[0] != state:
                auth_code["error"] = "State mismatch"
            elif "code" in qs:
                auth_code["code"] = qs["code"][0]
            elif "error" in qs:
                auth_code["error"] = qs.get("error_description", qs["error"])[0]
            else:
                auth_code["error"] = "No code in callback"

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            if auth_code["code"]:
                self.wfile.write(b"""<html><body style="font-family:system-ui;text-align:center;padding:60px;background:#1DB954;color:white">
                    <h1>&#x2705; Auth successful!</h1><p>You can close this tab and return to DevDuck.</p></body></html>""")
            else:
                self.wfile.write(f"<h1>&#x274c; Error: {auth_code['error']}</h1>".encode())

        def log_message(self, *args):
            pass  # Suppress logs

    # Bind to the right host (127.0.0.1 or localhost based on redirect_uri)
    parsed_ruri = urllib.parse.urlparse(redirect_uri)
    bind_host = parsed_ruri.hostname or "127.0.0.1"

    server = HTTPServer((bind_host, port), CallbackHandler)
    server.timeout = 120

    # Open browser
    print(f"  🌐 Opening browser for OAuth2 authorization...")
    print(f"  📋 Redirect: {redirect_uri}")
    webbrowser.open(full_auth_url)

    # Wait for callback
    print(f"  ⏳ Waiting for callback on {bind_host}:{port}...")
    server.handle_request()
    server.server_close()

    if auth_code["error"]:
        return {"error": auth_code["error"]}
    if not auth_code["code"]:
        return {"error": "No authorization code received"}

    # Exchange code for token
    # Some providers (Spotify) want Basic auth header instead of client_secret in body

    token_body = {
        "grant_type": "authorization_code",
        "code": auth_code["code"],
        "redirect_uri": redirect_uri,
    }

    # Build auth: try Basic auth header (Spotify-style) with client_id in body as fallback
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "Authorization": f"Basic {credentials}",
    }

    req = urllib.request.Request(
        token_url,
        data=urllib.parse.urlencode(token_body).encode(),
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            token_response = json.loads(resp.read().decode())
    except Exception:
        # Fallback: send client_id/secret in body (GitHub-style)
        token_body["client_id"] = client_id
        token_body["client_secret"] = client_secret
        req = urllib.request.Request(
            token_url,
            data=urllib.parse.urlencode(token_body).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            token_response = json.loads(resp.read().decode())

    result = {
        "access_token": token_response.get("access_token"),
        "refresh_token": token_response.get("refresh_token"),
        "token_type": token_response.get("token_type", "Bearer"),
        "expires_at": time.time() + token_response.get("expires_in", 3600),
        "scope": token_response.get("scope", " ".join(scopes)),
        "token_url": token_url,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }
    return result


def _do_oauth2_client_credentials(
    scheme: dict,
    client_id: str,
    client_secret: str,
    scopes: list = None,
) -> dict:
    """Perform OAuth2 client credentials flow."""
    flows = scheme.get("flows", {})
    cc_flow = flows.get("clientCredentials", {})
    token_url = cc_flow.get("tokenUrl", "")
    available_scopes = cc_flow.get("scopes", {})

    if not token_url:
        return {"error": "Missing tokenUrl in spec"}

    if scopes is None:
        scopes = list(available_scopes.keys())


    # Some APIs want Basic auth header, others want form body
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    token_data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": " ".join(scopes),
    }).encode()

    req = urllib.request.Request(
        token_url,
        data=token_data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {auth_header}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        token_response = json.loads(resp.read().decode())

    return {
        "access_token": token_response.get("access_token"),
        "token_type": token_response.get("token_type", "Bearer"),
        "expires_at": time.time() + token_response.get("expires_in", 3600),
        "scope": token_response.get("scope", " ".join(scopes)),
    }


def _refresh_oauth2_token(token_data: dict, token_url: str, client_id: str, client_secret: str) -> dict:
    """Refresh an OAuth2 token."""
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        return {"error": "No refresh token available"}


    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode()

    req = urllib.request.Request(
        token_url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {auth_header}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        new_token = json.loads(resp.read().decode())

    return {
        "access_token": new_token.get("access_token"),
        "refresh_token": new_token.get("refresh_token", refresh_token),
        "token_type": new_token.get("token_type", "Bearer"),
        "expires_at": time.time() + new_token.get("expires_in", 3600),
    }


def _apply_auth(req_headers: dict, req_params: dict, spec: dict, security_reqs: list, alias: str) -> tuple:
    """Apply authentication to a request based on security requirements.

    Auto-refreshes expired OAuth2 tokens when refresh_token + token_url are stored.
    """
    schemes = _extract_security_schemes(spec)
    token_data = _load_token(alias)

    # Auto-refresh expired OAuth2 tokens
    if token_data and token_data.get("type") == "oauth2" and _is_token_expired(token_data):
        refresh_token = token_data.get("refresh_token")
        token_url = token_data.get("token_url")
        cid = token_data.get("client_id")
        csec = token_data.get("client_secret")
        if refresh_token and token_url and cid:
            print(f"  🔄 Token expired for `{alias}`, auto-refreshing...")
            try:
                new_token = _refresh_oauth2_token(token_data, token_url, cid, csec or "")
                if "error" not in new_token:
                    # Preserve metadata from old token
                    new_token["type"] = "oauth2"
                    new_token["flow"] = token_data.get("flow", "authorization_code")
                    new_token["token_url"] = token_url
                    new_token["client_id"] = cid
                    new_token["client_secret"] = csec or ""
                    new_token["redirect_uri"] = token_data.get("redirect_uri", "")
                    _save_token(alias, new_token)
                    token_data = new_token
                    print(f"  ✅ Token refreshed, expires {time.ctime(new_token.get('expires_at', 0))}")
                else:
                    print(f"  ⚠️  Refresh failed: {new_token['error']}. Re-auth may be needed.")
            except Exception as e:
                print(f"  ⚠️  Refresh failed: {e}. Re-auth may be needed.")

    for sec_req in security_reqs:
        if not isinstance(sec_req, dict):
            continue
        for scheme_name, required_scopes in sec_req.items():
            scheme = schemes.get(scheme_name, {})
            scheme_type = scheme.get("type", "")

            if scheme_type == "oauth2" and token_data and token_data.get("access_token"):
                token_type = token_data.get("token_type", "Bearer")
                req_headers["Authorization"] = f"{token_type} {token_data['access_token']}"
                return req_headers, req_params

            elif scheme_type == "http":
                http_scheme = scheme.get("scheme", "bearer").lower()
                if http_scheme == "bearer" and token_data and token_data.get("access_token"):
                    req_headers["Authorization"] = f"Bearer {token_data['access_token']}"
                elif http_scheme == "basic" and token_data:
                    creds = f"{token_data.get('username', '')}:{token_data.get('password', '')}"
                    req_headers["Authorization"] = f"Basic {base64.b64encode(creds.encode()).decode()}"
                return req_headers, req_params

            elif scheme_type == "apiKey":
                key_name = scheme.get("name", "")
                key_in = scheme.get("in", "header")
                api_key = token_data.get("api_key", "") if token_data else os.getenv(f"OPENAPI_{alias.upper()}_API_KEY", "")
                if api_key:
                    if key_in == "header":
                        req_headers[key_name] = api_key
                    elif key_in == "query":
                        req_params[key_name] = api_key
                    elif key_in == "cookie":
                        req_headers["Cookie"] = f"{key_name}={api_key}"
                return req_headers, req_params

    # No security matched — check if we have a generic bearer token
    if token_data and token_data.get("access_token"):
        req_headers["Authorization"] = f"Bearer {token_data['access_token']}"
    elif token_data and token_data.get("api_key"):
        req_headers["Authorization"] = f"Bearer {token_data['api_key']}"

    return req_headers, req_params


def _execute_request(
    method: str,
    url: str,
    headers: dict = None,
    params: dict = None,
    body: Any = None,
    timeout: int = 30,
) -> dict:
    """Execute an HTTP request and return the response."""

    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    data = None
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            headers = headers or {}
            headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        elif isinstance(body, bytes):
            data = body

    headers = headers or {}
    headers.setdefault("Accept", "application/json")
    headers.setdefault("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                try:
                    resp_body = json.loads(resp_body)
                except json.JSONDecodeError:
                    pass
            return {
                "status": resp.status,
                "headers": dict(resp.headers),
                "body": resp_body,
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return {
            "status": e.code,
            "error": e.reason,
            "body": error_body,
        }
    except Exception as e:
        return {"status": 0, "error": str(e)}


@tool
def openapi(
    action: str = "help",
    spec_url: str = None,
    alias: str = None,
    operation: str = None,
    params: str = None,
    body: str = None,
    method: str = None,
    path: str = None,
    client_id: str = None,
    client_secret: str = None,
    api_key: str = None,
    token: str = None,
    username: str = None,
    password: str = None,
    scopes: str = None,
    auth_flow: str = None,
    redirect_uri: str = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """🔌 Universal OpenAPI client — load any API spec and call endpoints with auto-auth.

    Supports OAuth2 (all flows), API keys, Bearer tokens, Basic auth.
    Reads OpenAPI 3.x and Swagger 2.0 specs.

    Args:
        action: Action to perform:
            - "load": Load an OpenAPI spec from URL
            - "list": List loaded specs and their operations
            - "call": Call an API operation by operationId
            - "raw": Make a raw request (method + path)
            - "auth": Configure authentication for a spec
            - "token": View/manage stored tokens
            - "schemas": Show schemas/models from a spec
            - "help": Show usage help
        spec_url: URL or local path to OpenAPI/Swagger spec — JSON or YAML (for load)
        alias: Short name for the API (auto-generated from spec title if omitted)
        operation: operationId to call (for call action)
        params: JSON string of query/path parameters
        body: JSON string of request body
        method: HTTP method for raw requests
        path: URL path for raw requests
        client_id: OAuth2 client ID (for auth)
        client_secret: OAuth2 client secret (for auth)
        api_key: API key value (for auth)
        token: Bearer token value (for auth)
        username: Basic auth username (for auth)
        password: Basic auth password (for auth)
        scopes: Comma-separated OAuth2 scopes (for auth)
        auth_flow: OAuth2 flow type: "authorization_code", "client_credentials", "password" (for auth)
        timeout: Request timeout in seconds

    Returns:
        Dict with status and response content

    Examples:
        # Load a JSON API spec
        openapi(action="load", spec_url="https://petstore3.swagger.io/api/v3/openapi.json")

        # Load a YAML spec (from URL)
        openapi(action="load", spec_url="https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.yaml")

        # Load a YAML spec (from local file)
        openapi(action="load", spec_url="./my-api-spec.yaml")

        # Load from GitHub blob URL (auto-converts to raw)
        openapi(action="load", spec_url="https://github.com/owner/repo/blob/main/openapi.yml")

        # List operations
        openapi(action="list", alias="petstore")

        # Call an operation
        openapi(action="call", alias="petstore", operation="findPetsByStatus", params='{"status": "available"}')

        # Set API key
        openapi(action="auth", alias="myapi", api_key="sk-xxx")

        # OAuth2 authorization code flow (opens browser)
        openapi(action="auth", alias="github", auth_flow="authorization_code",
                client_id="xxx", client_secret="yyy", scopes="read:user,repo")

        # OAuth2 client credentials
        openapi(action="auth", alias="myapi", auth_flow="client_credentials",
                client_id="xxx", client_secret="yyy")
    """
    try:
        parsed_params = json.loads(params) if params else {}
        parsed_body = json.loads(body) if body else None
        scope_list = [s.strip() for s in scopes.split(",")] if scopes else None
    except json.JSONDecodeError as e:
        return {"status": "error", "content": [{"text": f"Invalid JSON: {e}"}]}

    # ── LOAD ──
    if action == "load":
        if not spec_url:
            return {"status": "error", "content": [{"text": "spec_url is required for load action"}]}

        try:
            spec = _fetch_spec(spec_url)
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to fetch spec: {e}"}]}

        # Determine alias
        title = spec.get("info", {}).get("title", "api")
        auto_alias = alias or title.lower().replace(" ", "_").replace("-", "_")[:20]
        auto_alias = "".join(c for c in auto_alias if c.isalnum() or c == "_")

        base_url = _get_base_url(spec, spec_url)
        if not base_url:
            # Derive from spec_url
            parsed = urllib.parse.urlparse(spec_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

        _LOADED_SPECS[auto_alias] = spec
        _BASE_URLS[auto_alias] = base_url

        # Cache spec to disk
        cache_file = _STORE_DIR / f"spec_{auto_alias}.json"
        try:
            cache_file.write_text(json.dumps({"spec_url": spec_url, "base_url": base_url, "spec": spec}, default=str))
        except Exception:
            pass  # Cache failure is non-fatal

        operations = _extract_operations(spec)
        security_schemes = _extract_security_schemes(spec)

        # Build summary
        lines = [
            f"✅ Loaded: **{title}** as `{auto_alias}`",
            f"   Base URL: {base_url}",
            f"   Operations: {len(operations)}",
            f"   Security schemes: {list(security_schemes.keys()) if security_schemes else 'none'}",
            "",
        ]

        for op_id, op in sorted(operations.items()):
            sec_tag = " 🔒" if op["security"] else ""
            lines.append(f"  {op['method']:7} {op['path']:40} {op_id}{sec_tag}")
            if op["summary"]:
                lines.append(f"          {op['summary'][:80]}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # ── LIST ──
    elif action == "list":
        if not _LOADED_SPECS:
            # Try loading from disk cache
            for cache_file in _STORE_DIR.glob("spec_*.json"):
                try:
                    cached = json.loads(cache_file.read_text())
                    a = cache_file.stem.replace("spec_", "")
                    _LOADED_SPECS[a] = cached["spec"]
                    _BASE_URLS[a] = cached["base_url"]
                except Exception:
                    pass

        if not _LOADED_SPECS:
            return {"status": "success", "content": [{"text": "No APIs loaded. Use openapi(action='load', spec_url='...')"}]}

        lines = ["📚 Loaded APIs:\n"]
        for a, spec in _LOADED_SPECS.items():
            if alias and alias != a:
                continue
            title = spec.get("info", {}).get("title", "?")
            ops = _extract_operations(spec)
            security = _extract_security_schemes(spec)
            token = _load_token(a)
            auth_status = "✅ authenticated" if token and token.get("access_token", token.get("api_key")) else "🔓 no auth"

            lines.append(f"**{a}** — {title}")
            lines.append(f"  Base: {_BASE_URLS.get(a, '?')} | Auth: {auth_status}")
            lines.append(f"  Operations ({len(ops)}):")
            for op_id, op in sorted(ops.items()):
                sec_tag = " 🔒" if op["security"] else ""
                lines.append(f"    {op['method']:7} {op['path']:40} `{op_id}`{sec_tag}")
            lines.append("")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # ── AUTH ──
    elif action == "auth":
        if not alias:
            return {"status": "error", "content": [{"text": "alias is required — which API to authenticate?"}]}

        # Simple API key
        if api_key:
            _save_token(alias, {"api_key": api_key, "type": "api_key"})
            return {"status": "success", "content": [{"text": f"✅ API key stored for `{alias}`"}]}

        # Simple bearer token
        if token:
            _save_token(alias, {"access_token": token, "token_type": "Bearer", "type": "bearer"})
            return {"status": "success", "content": [{"text": f"✅ Bearer token stored for `{alias}`"}]}

        # Basic auth
        if username and password:
            _save_token(alias, {"username": username, "password": password, "type": "basic"})
            return {"status": "success", "content": [{"text": f"✅ Basic auth stored for `{alias}`"}]}

        # OAuth2
        if auth_flow:
            # Auto-detect client_id/secret from env vars if not provided
            effective_client_id = client_id
            effective_client_secret = client_secret
            if not effective_client_id:
                for env_key in [
                    f"{alias.upper()}_CLIENT_ID",
                    f"{alias.upper().split('_')[0]}_CLIENT_ID",
                ]:
                    val = os.environ.get(env_key)
                    if val:
                        effective_client_id = val
                        print(f"  🔍 Auto-detected client_id from ${env_key}")
                        break
            if not effective_client_secret:
                for env_key in [
                    f"{alias.upper()}_CLIENT_SECRET",
                    f"{alias.upper().split('_')[0]}_CLIENT_SECRET",
                ]:
                    val = os.environ.get(env_key)
                    if val:
                        effective_client_secret = val
                        print(f"  🔍 Auto-detected client_secret from ${env_key}")
                        break

            if not effective_client_id:
                return {"status": "error", "content": [{"text": f"client_id required. Set it directly or via ${alias.upper()}_CLIENT_ID env var."}]}

            spec = _LOADED_SPECS.get(alias)
            if not spec:
                return {"status": "error", "content": [{"text": f"Spec `{alias}` not loaded. Load it first."}]}

            schemes = _extract_security_schemes(spec)
            oauth_scheme = None
            for name, scheme in schemes.items():
                if scheme.get("type") == "oauth2":
                    oauth_scheme = scheme
                    break

            if not oauth_scheme:
                return {"status": "error", "content": [{"text": f"No OAuth2 scheme found in `{alias}` spec"}]}

            # Auto-detect redirect_uri from env vars (e.g. SPOTIFY_REDIRECT_URI)
            effective_redirect_uri = redirect_uri
            if not effective_redirect_uri:
                # Try common env var patterns: {ALIAS}_REDIRECT_URI, {ALIAS}_CALLBACK_URL
                for env_key in [
                    f"{alias.upper()}_REDIRECT_URI",
                    f"{alias.upper()}_CALLBACK_URL",
                    f"{alias.upper().split('_')[0]}_REDIRECT_URI",
                ]:
                    val = os.environ.get(env_key)
                    if val:
                        effective_redirect_uri = val
                        print(f"  🔍 Auto-detected redirect_uri from ${env_key}")
                        break

            if auth_flow == "authorization_code":
                result = _do_oauth2_authorization_code(
                    oauth_scheme, effective_client_id, effective_client_secret or "",
                    scope_list, redirect_uri=effective_redirect_uri,
                )
            elif auth_flow == "client_credentials":
                result = _do_oauth2_client_credentials(
                    oauth_scheme, effective_client_id, effective_client_secret or "", scope_list
                )
            elif auth_flow == "password":
                # Resource owner password flow
                flows = oauth_scheme.get("flows", {})
                pw_flow = flows.get("password", {})
                token_url = pw_flow.get("tokenUrl", "")
                if not token_url:
                    return {"status": "error", "content": [{"text": "No tokenUrl for password flow"}]}

                body_data = urllib.parse.urlencode({
                    "grant_type": "password",
                    "username": username or "",
                    "password": password or "",
                    "client_id": effective_client_id,
                    "scope": " ".join(scope_list) if scope_list else "",
                }).encode()
                req = urllib.request.Request(token_url, data=body_data, headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    token_resp = json.loads(resp.read().decode())
                result = {
                    "access_token": token_resp.get("access_token"),
                    "refresh_token": token_resp.get("refresh_token"),
                    "token_type": token_resp.get("token_type", "Bearer"),
                    "expires_at": time.time() + token_resp.get("expires_in", 3600),
                }
            else:
                return {"status": "error", "content": [{"text": f"Unknown auth_flow: {auth_flow}. Use: authorization_code, client_credentials, password"}]}

            if "error" in result:
                return {"status": "error", "content": [{"text": f"OAuth2 failed: {result['error']}"}]}

            result["type"] = "oauth2"
            result["flow"] = auth_flow
            _save_token(alias, result)
            return {"status": "success", "content": [{"text": f"✅ OAuth2 ({auth_flow}) token obtained for `{alias}`\n   Expires: {time.ctime(result.get('expires_at', 0))}"}]}

        return {"status": "error", "content": [{"text": "Provide one of: api_key, token, username+password, or auth_flow+client_id"}]}

    # ── TOKEN ──
    elif action == "token":
        if alias:
            token_data = _load_token(alias)
            if not token_data:
                return {"status": "success", "content": [{"text": f"No token stored for `{alias}`"}]}
            # Redact sensitive values for display
            display = dict(token_data)
            if "access_token" in display:
                display["access_token"] = display["access_token"][:8] + "..." + display["access_token"][-4:]
            if "api_key" in display:
                display["api_key"] = display["api_key"][:8] + "..." + display["api_key"][-4:]
            if "refresh_token" in display:
                display["refresh_token"] = "***"
            if "password" in display:
                display["password"] = "***"
            expired = _is_token_expired(token_data)
            display["expired"] = expired
            return {"status": "success", "content": [{"text": f"🔑 Token for `{alias}`:\n{json.dumps(display, indent=2)}"}]}
        else:
            # List all tokens
            tokens = []
            for tf in _STORE_DIR.glob("token_*.json"):
                name = tf.stem.replace("token_", "")
                data = json.loads(tf.read_text())
                auth_type = data.get("type", "unknown")
                expired = _is_token_expired(data)
                tokens.append(f"  {'🔴' if expired else '🟢'} `{name}` — {auth_type}" + (" (expired)" if expired else ""))
            if not tokens:
                return {"status": "success", "content": [{"text": "No stored tokens."}]}
            return {"status": "success", "content": [{"text": "🔑 Stored tokens:\n" + "\n".join(tokens)}]}

    # ── CALL ──
    elif action == "call":
        if not operation:
            return {"status": "error", "content": [{"text": "operation (operationId) is required"}]}

        # Find the operation across loaded specs
        target_spec = None
        target_alias = alias
        target_op = None

        if alias and alias in _LOADED_SPECS:
            ops = _extract_operations(_LOADED_SPECS[alias])
            if operation in ops:
                target_spec = _LOADED_SPECS[alias]
                target_op = ops[operation]
            else:
                return {"status": "error", "content": [{"text": f"Operation `{operation}` not found in `{alias}`. Use action='list' to see available operations."}]}
        else:
            # Search all loaded specs
            for a, spec in _LOADED_SPECS.items():
                ops = _extract_operations(spec)
                if operation in ops:
                    target_spec = spec
                    target_alias = a
                    target_op = ops[operation]
                    break

        if not target_op:
            return {"status": "error", "content": [{"text": f"Operation `{operation}` not found. Load a spec first or check the operationId."}]}

        # Build URL
        base = _BASE_URLS.get(target_alias, "")
        url_path = target_op["path"]

        # Substitute path parameters
        query_params = {}
        for param in target_op.get("parameters", []):
            p_name = param.get("name", "")
            p_in = param.get("in", "")
            p_value = parsed_params.get(p_name)
            if p_value is not None:
                if p_in == "path":
                    url_path = url_path.replace(f"{{{p_name}}}", str(p_value))
                elif p_in == "query":
                    query_params[p_name] = p_value
            elif param.get("required") and p_in == "path":
                return {"status": "error", "content": [{"text": f"Required path parameter missing: {p_name}"}]}

        # Add any extra params as query params
        for k, v in parsed_params.items():
            if k not in query_params and f"{{{k}}}" not in target_op["path"]:
                query_params[k] = v

        full_url = f"{base}{url_path}"
        headers = {}

        # Apply auth
        headers, query_params = _apply_auth(headers, query_params, target_spec, target_op.get("security", []), target_alias)

        # Handle request body
        req_body = parsed_body
        if req_body is None and target_op.get("requestBody"):
            # Check if params should be body (for POST/PUT/PATCH without explicit body)
            if target_op["method"] in ("POST", "PUT", "PATCH") and parsed_params:
                req_body = parsed_params
                query_params = {}  # Don't send as query params

        # Execute
        result = _execute_request(
            method=target_op["method"],
            url=full_url,
            headers=headers,
            params=query_params if query_params else None,
            body=req_body,
            timeout=timeout,
        )

        response_text = f"**{target_op['method']} {url_path}** → {result.get('status')}\n"
        body_content = result.get("body", "")
        if isinstance(body_content, dict):
            response_text += json.dumps(body_content, indent=2, ensure_ascii=False)
        else:
            response_text += str(body_content)[:5000]

        return {"status": "success" if result.get("status", 0) < 400 else "error", "content": [{"text": response_text}]}

    # ── RAW ──
    elif action == "raw":
        if not method or not path:
            return {"status": "error", "content": [{"text": "method and path required for raw action"}]}

        if alias and alias in _BASE_URLS:
            full_url = f"{_BASE_URLS[alias]}{path}"
        elif path.startswith("http"):
            full_url = path
        else:
            return {"status": "error", "content": [{"text": "Provide alias (with loaded spec) or full URL in path"}]}

        headers = {}
        if alias:
            spec = _LOADED_SPECS.get(alias, {})
            headers, parsed_params = _apply_auth(headers, parsed_params, spec, spec.get("security", []), alias)

        result = _execute_request(
            method=method.upper(),
            url=full_url,
            headers=headers,
            params=parsed_params if parsed_params else None,
            body=parsed_body,
            timeout=timeout,
        )

        response_text = f"**{method.upper()} {path}** → {result.get('status')}\n"
        body_content = result.get("body", "")
        if isinstance(body_content, dict):
            response_text += json.dumps(body_content, indent=2, ensure_ascii=False)
        else:
            response_text += str(body_content)[:5000]

        return {"status": "success" if result.get("status", 0) < 400 else "error", "content": [{"text": response_text}]}

    # ── SCHEMAS ──
    elif action == "schemas":
        if not alias or alias not in _LOADED_SPECS:
            return {"status": "error", "content": [{"text": "alias required — load a spec first"}]}

        spec = _LOADED_SPECS[alias]
        schemas = spec.get("components", {}).get("schemas", spec.get("definitions", {}))

        if not schemas:
            return {"status": "success", "content": [{"text": f"No schemas found in `{alias}`"}]}

        lines = [f"📋 Schemas in `{alias}`:\n"]
        for name, schema in sorted(schemas.items()):
            desc = schema.get("description", "")[:80]
            schema_type = schema.get("type", "object")
            props = schema.get("properties", {})
            lines.append(f"  **{name}** ({schema_type}) — {desc}")
            for prop_name, prop_schema in list(props.items())[:10]:
                p_type = prop_schema.get("type", prop_schema.get("$ref", "?"))
                required = prop_name in schema.get("required", [])
                req_tag = " *" if required else ""
                lines.append(f"    .{prop_name}: {p_type}{req_tag}")
            if len(props) > 10:
                lines.append(f"    ... +{len(props) - 10} more")
            lines.append("")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # ── HELP ──
    else:
        return {"status": "success", "content": [{"text": """🔌 **OpenAPI Tool** — Universal API client

**Quick Start:**
```
openapi(action="load", spec_url="https://api.example.com/openapi.json")
openapi(action="list")
openapi(action="call", operation="getUser", params='{"id": 123}')
```

**Actions:**
- `load` — Load OpenAPI spec from URL
- `list` — Show loaded APIs and operations
- `call` — Call an operation by operationId
- `raw` — Raw HTTP request (method + path)
- `auth` — Configure auth (api_key, token, oauth2)
- `token` — View stored tokens
- `schemas` — Show API schemas
- `help` — This help

**Auth Examples:**
```
# API Key
openapi(action="auth", alias="myapi", api_key="sk-xxx")

# Bearer Token
openapi(action="auth", alias="myapi", token="eyJ...")

# OAuth2 Authorization Code (opens browser)
openapi(action="auth", alias="github", auth_flow="authorization_code",
        client_id="xxx", client_secret="yyy")

# OAuth2 Client Credentials (server-to-server)
openapi(action="auth", alias="stripe", auth_flow="client_credentials",
        client_id="xxx", client_secret="yyy")
```

**Tokens are persisted** in `~/.devduck/openapi/` and auto-applied to matching requests.
"""}]}


