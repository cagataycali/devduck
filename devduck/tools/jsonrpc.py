"""
🔗 Generic JSON-RPC 2.0 Client Tool

Universal JSON-RPC client supporting HTTP and WebSocket transports with
secure authentication via environment variables.
"""

import json
import os
import time
import asyncio
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import urllib.request
import urllib.error

from strands import tool


def _make_http_request(
    endpoint: str,
    method: str,
    params: List = None,
    headers: Dict[str, str] = None,
    request_id: int = 1,
    timeout: int = 30,
    jsonrpc_version: str = "2.0",
) -> Dict[str, Any]:
    """Make JSON-RPC request over HTTP."""
    headers = headers or {}
    headers.setdefault("Content-Type", "application/json")

    payload = {
        "jsonrpc": jsonrpc_version,
        "id": request_id,
        "method": method,
        "params": params or [],
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start_time = time.time()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            duration = time.time() - start_time

            return {
                "success": "error" not in result,
                "result": result.get("result"),
                "error": result.get("error"),
                "duration": duration,
                "transport": "http",
            }

    except urllib.error.HTTPError as e:
        duration = time.time() - start_time
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            return {
                "success": False,
                "error": error_body.get("error", {"code": e.code, "message": str(e)}),
                "duration": duration,
                "transport": "http",
            }
        except Exception:
            return {
                "success": False,
                "error": {"code": e.code, "message": f"HTTP {e.code}: {e.reason}"},
                "duration": duration,
                "transport": "http",
            }

    except urllib.error.URLError as e:
        return {
            "success": False,
            "error": {"code": -1, "message": f"Connection error: {e.reason}"},
            "duration": time.time() - start_time,
            "transport": "http",
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": {"code": -2, "message": f"Invalid JSON response: {e}"},
            "duration": time.time() - start_time,
            "transport": "http",
        }


def _make_ws_request(
    endpoint: str,
    method: str,
    params: List = None,
    headers: Dict[str, str] = None,
    request_id: int = 1,
    timeout: int = 30,
    jsonrpc_version: str = "2.0",
) -> Dict[str, Any]:
    """Make JSON-RPC request over WebSocket."""
    try:
        import websockets
    except ImportError:
        return {
            "success": False,
            "error": {
                "code": -3,
                "message": "websockets package required. Install: pip install websockets",
            },
            "duration": 0,
            "transport": "websocket",
        }

    payload = {
        "jsonrpc": jsonrpc_version,
        "id": request_id,
        "method": method,
        "params": params or [],
    }

    async def _ws_call():
        start_time = time.time()
        try:
            # Build extra headers for connection
            extra_headers = headers if headers else {}

            async with websockets.connect(
                endpoint,
                additional_headers=extra_headers,
                open_timeout=timeout,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps(payload))
                response = await asyncio.wait_for(ws.recv(), timeout=timeout)
                result = json.loads(response)
                duration = time.time() - start_time

                return {
                    "success": "error" not in result,
                    "result": result.get("result"),
                    "error": result.get("error"),
                    "duration": duration,
                    "transport": "websocket",
                }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": {"code": -4, "message": f"WebSocket timeout after {timeout}s"},
                "duration": time.time() - start_time,
                "transport": "websocket",
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"code": -5, "message": f"WebSocket error: {str(e)}"},
                "duration": time.time() - start_time,
                "transport": "websocket",
            }

    # Run async function
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # If we're already in an async context, create a new loop in a thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _ws_call())
            return future.result(timeout=timeout + 5)
    else:
        return loop.run_until_complete(_ws_call())


def _process_auth(
    headers: Dict[str, str],
    auth_type: Optional[str],
    auth_token: Optional[str],
    auth_env_var: Optional[str],
    auth_header: Optional[str],
) -> Dict[str, str]:
    """Process authentication and add to headers."""
    # Get token from env var if specified
    token = auth_token

    if auth_env_var:
        env_token = os.getenv(auth_env_var)
        if env_token:
            token = env_token
        else:
            raise ValueError(
                f"Environment variable '{auth_env_var}' not found or empty."
            )

    if not token:
        return headers

    # Determine header name
    header_name = auth_header or "Authorization"

    # Apply auth based on type
    if auth_type == "Bearer":
        headers[header_name] = f"Bearer {token}"
    elif auth_type == "token":
        headers[header_name] = f"token {token}"
    elif auth_type == "api_key":
        if header_name == "Authorization":
            headers["X-API-Key"] = token
        else:
            headers[header_name] = token
    elif auth_type == "basic":
        import base64

        encoded = base64.b64encode(token.encode()).decode()
        headers[header_name] = f"Basic {encoded}"
    else:
        # Default: add token directly
        headers[header_name] = token

    return headers


@tool
def jsonrpc(
    method: str,
    params: List = None,
    endpoint: str = None,
    id: int = 1,
    timeout: int = 30,
    jsonrpc_version: str = "2.0",
    auth_type: str = None,
    auth_env_var: str = None,
    auth_token: str = None,
    auth_header: str = None,
    headers: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Generic JSON-RPC client for any RPC service.

    Supports both HTTP (http://, https://) and WebSocket (ws://, wss://) transports.
    Transport is auto-detected from URL scheme.

    Args:
        method: RPC method name (e.g., "getInfo", "eth_blockNumber")
        params: List of parameters for the method (default: [])
        endpoint: RPC endpoint URL - HTTP or WebSocket
        id: JSON-RPC request ID (default: 1)
        timeout: Request timeout in seconds (default: 30)
        jsonrpc_version: JSON-RPC version (default: "2.0")
        auth_type: Authentication type (Bearer, token, api_key, basic, custom)
        auth_env_var: Environment variable containing auth token (RECOMMENDED)
        auth_token: Direct auth token (prefer auth_env_var for security)
        auth_header: Custom auth header name (default: Authorization)
        headers: Additional HTTP headers as key-value pairs

    Returns:
        Dict with status and RPC response

    Environment Variables:
        JSONRPC_ENDPOINT: Default endpoint if none specified

    Examples:
        # HTTP endpoint
        jsonrpc(method="getInfo", endpoint="https://my-server.com/rpc")

        # WebSocket endpoint
        jsonrpc(method="subscribe", endpoint="wss://my-server.com/ws")

        # With auth from environment (SECURE - never hardcode tokens!)
        jsonrpc(
            method="getPrivateData",
            endpoint="https://api.service.com/rpc",
            auth_env_var="SERVICE_API_KEY",
            auth_type="Bearer"
        )

        # With params
        jsonrpc(
            method="getData",
            params=["arg1", {"key": "value"}],
            endpoint="https://api.example.com/rpc"
        )

        # Custom headers
        jsonrpc(
            method="getData",
            endpoint="https://api.example.com/rpc",
            headers={"X-Custom-Header": "value"}
        )
    """
    try:
        # Determine endpoint
        url = endpoint or os.getenv("JSONRPC_ENDPOINT")

        if not url:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "No endpoint specified. Provide 'endpoint' URL or set JSONRPC_ENDPOINT env var."
                    }
                ],
            }

        # Process headers and auth
        request_headers = dict(headers) if headers else {}

        try:
            request_headers = _process_auth(
                request_headers, auth_type, auth_token, auth_env_var, auth_header
            )
        except ValueError as e:
            return {"status": "error", "content": [{"text": str(e)}]}

        # Determine transport based on URL scheme
        parsed = urlparse(url)

        if parsed.scheme in ("ws", "wss"):
            # WebSocket transport
            result = _make_ws_request(
                endpoint=url,
                method=method,
                params=params,
                headers=request_headers,
                request_id=id,
                timeout=timeout,
                jsonrpc_version=jsonrpc_version,
            )
        elif parsed.scheme in ("http", "https"):
            # HTTP transport
            result = _make_http_request(
                endpoint=url,
                method=method,
                params=params,
                headers=request_headers,
                request_id=id,
                timeout=timeout,
                jsonrpc_version=jsonrpc_version,
            )
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unsupported URL scheme: {parsed.scheme}. Use http(s) or ws(s)."
                    }
                ],
            }

        # Handle errors
        if not result["success"]:
            error = result.get("error", {})
            error_msg = f"RPC Error {error.get('code', '?')}: {error.get('message', 'Unknown error')}"
            return {
                "status": "error",
                "content": [{"text": error_msg}, {"json": result}],
            }

        # Format result
        raw_result = result.get("result")

        # Pretty format based on result type
        if isinstance(raw_result, dict):
            text = f"Result:\n```json\n{json.dumps(raw_result, indent=2)[:3000]}\n```"
        elif isinstance(raw_result, list):
            text = f"Result: {len(raw_result)} items\n```json\n{json.dumps(raw_result[:10], indent=2)[:2000]}\n```"
        elif isinstance(raw_result, str) and raw_result.startswith("0x"):
            try:
                decoded = int(raw_result, 16)
                text = f"Result: {raw_result} (decimal: {decoded:,})"
            except ValueError:
                text = f"Result: {raw_result}"
        else:
            text = f"Result: {raw_result}"

        return {
            "status": "success",
            "content": [
                {"text": text},
                {
                    "json": (
                        {
                            "result": raw_result,
                            "endpoint": url,
                            "method": method,
                            "duration": result.get("duration"),
                            "transport": result.get("transport"),
                        }
                    )
                },
            ],
        }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
