"""Browser Bridge - Control Chrome via WebSocket extension"""
from strands import tool
import asyncio
import json
import threading
from typing import Dict, Any, Optional
import uuid
import base64
import time
from queue import Queue

# WebSocket server state
_server = None
_clients = []
_server_thread = None
_running = False
_response_queues: Dict[str, Queue] = {}  # message_id -> response queue
_event_loop = None  # Server's event loop


async def handle_client(websocket):
    """Handle WebSocket client connection"""
    global _clients
    _clients.append(websocket)
    print(f"🦆 Browser connected: {len(_clients)} active")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get('type')
            msg_id = data.get('id')
            
            print(f"🦆 Browser message: type={msg_type}, id={msg_id}")
            
            if msg_type == 'hello':
                await websocket.send(json.dumps({
                    'type': 'welcome',
                    'data': {'server': 'devduck', 'version': '1.0.0'}
                }))
            
            elif msg_type == 'response' and msg_id:
                # Response to a command we sent
                if msg_id in _response_queues:
                    _response_queues[msg_id].put(data.get('data'))
            
            elif msg_type == 'error' and msg_id:
                # Error response
                if msg_id in _response_queues:
                    _response_queues[msg_id].put({'error': data.get('error')})
                    
    except Exception as e:
        print(f"🦆 Browser error: {e}")
    finally:
        if websocket in _clients:
            _clients.remove(websocket)
        print(f"🦆 Browser disconnected: {len(_clients)} active")


async def start_server_async(port=9223):
    """Start WebSocket server"""
    import websockets
    
    global _server, _running, _event_loop
    _running = True
    _event_loop = asyncio.get_event_loop()  # Store the loop
    
    _server = await websockets.serve(handle_client, "0.0.0.0", port)
    print(f"🦆 Browser bridge server started on ws://localhost:{port}")
    
    await asyncio.Future()  # Run forever


def start_server_thread(port=9223):
    """Start server in background thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_server_async(port))


def send_command_sync(command: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """Send command to browser and wait for response (synchronous)"""
    if not _clients:
        raise Exception("No browser connected")
    
    if not _event_loop:
        raise Exception("Server not running")
    
    message_id = str(uuid.uuid4())
    message = {
        'id': message_id,
        'command': command,
        'params': params
    }
    
    # Create response queue
    _response_queues[message_id] = Queue()
    
    try:
        # Send to first connected browser
        client = _clients[0]
        
        # Schedule send on the server's event loop (thread-safe)
        future = asyncio.run_coroutine_threadsafe(
            client.send(json.dumps(message)),
            _event_loop
        )
        # Wait for send to complete
        future.result(timeout=5)
        
        # Wait for response with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = _response_queues[message_id].get(timeout=0.5)
                return response
            except:
                continue
        
        raise Exception(f"Response timeout after {timeout}s")
        
    finally:
        # Cleanup
        if message_id in _response_queues:
            del _response_queues[message_id]


@tool
def browser_bridge(
    action: str,
    command: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    port: int = 9223,
) -> Dict[str, Any]:
    """
    Control Chrome browser via WebSocket extension.
    
    This tool enables bidirectional communication between DevDuck and Chrome
    through a WebSocket-based browser extension. No special Chrome flags needed.
    
    Actions:
        start_server: Start WebSocket server for extension connections
        stop_server: Stop the WebSocket server
        status: Check server and connection status
        send: Send command to connected browser
    
    Available Browser Commands (when action="send"):
        Navigation:
        - navigate: Go to URL {"url": "https://example.com", "tabId": 123}
        - goBack: Go back {"tabId": 123}
        - goForward: Go forward {"tabId": 123}
        - reloadTab: Reload tab {"tabId": 123}
        - getCurrentUrl: Get current URL {"tabId": 123}
        
        Tab Management:
        - getTabs: List all open tabs {}
        - newTab: Open new tab {"url": "https://example.com"}
        - selectTab: Switch to tab {"tabId": 123}
        - closeTab: Close specific tab {"tabId": 123}
        
        Page Interaction:
        - execute: Run JavaScript {"code": "document.title", "tabId": 123}
        - click: Click element {"selector": "#button", "tabId": 123}
        - fill: Fill input {"selector": "input[name='q']", "value": "search", "tabId": 123}
        - waitForSelector: Wait for element {"selector": "#el", "timeout": 5000, "tabId": 123}
        
        Content Extraction:
        - getContent: Get page HTML/text/title/url {"tabId": 123}
        - screenshot: Capture visible tab {"tabId": 123}
        - getConsoleLogs: Capture console logs {"tabId": 123}
        
        System Integration:
        - readClipboard: Read from clipboard {}
        - writeClipboard: Write to clipboard {"text": "content"}
        - getLocation: Get geolocation {"tabId": 123}
        - listUSB: List USB devices {"tabId": 123}
        - showNotification: Show notification {"title": "Title", "message": "Message", "icon": "icon.png"}
        
        Storage:
        - getStorage: Get localStorage/sessionStorage {"tabId": 123, "type": "local"}
        - setStorage: Set storage value {"tabId": 123, "type": "local", "key": "mykey", "value": "myvalue"}
    
    Args:
        action: Operation to perform (start_server, stop_server, status, send)
        command: Browser command name (required when action="send")
        params: Command parameters dict (optional, defaults to {})
        port: WebSocket server port (default: 9223)
    
    Returns:
        Dict with status and content
    
    Examples:
        # Start server
        browser_bridge(action="start_server")
        
        # Check status
        browser_bridge(action="status")
        
        # Navigate to URL
        browser_bridge(
            action="send",
            command="navigate",
            params={"url": "https://example.com"}
        )
        
        # Execute JavaScript
        browser_bridge(
            action="send",
            command="execute",
            params={"code": "document.title"}
        )
        
        # Take screenshot
        browser_bridge(action="send", command="screenshot")
        
        # Click element
        browser_bridge(
            action="send",
            command="click",
            params={"selector": "#submit-button"}
        )
        
        # Read clipboard
        browser_bridge(action="send", command="readClipboard")
        
        # Write to clipboard
        browser_bridge(
            action="send",
            command="writeClipboard",
            params={"text": "DevDuck was here!"}
        )
        
        # Show notification
        browser_bridge(
            action="send",
            command="showNotification",
            params={"title": "DevDuck", "message": "Task complete!"}
        )
        
        # Get localStorage
        browser_bridge(
            action="send",
            command="getStorage",
            params={"type": "local"}
        )
    
    Setup:
        1. Load extension: chrome://extensions/ → Load unpacked
        2. Select: ~/devduck/extensions/chrome
        3. Extension auto-connects to ws://localhost:9223
        4. Green badge = connected, Red badge = disconnected
    """
    global _server_thread, _running
    
    try:
        if action == "start_server":
            if _server_thread and _running:
                return {
                    "status": "success",
                    "content": [{"text": f"✓ Browser bridge already running on port {port}"}]
                }
            
            _server_thread = threading.Thread(
                target=start_server_thread,
                args=(port,),
                daemon=True
            )
            _server_thread.start()
            
            return {
                "status": "success",
                "content": [{"text": f"✓ Browser bridge started on ws://localhost:{port}\n\nLoad Chrome extension from ~/devduck/extensions/chrome"}]
            }
        
        elif action == "stop_server":
            _running = False
            return {
                "status": "success",
                "content": [{"text": "✓ Browser bridge stopped"}]
            }
        
        elif action == "status":
            status_text = f"""Browser Bridge Status:
Running: {_running}
Port: {port}
Connected browsers: {len(_clients)}

Extension: ~/devduck/extensions/chrome
Load at: chrome://extensions/"""
            
            return {
                "status": "success",
                "content": [{"text": status_text}]
            }
        
        elif action == "send":
            if not command:
                return {
                    "status": "error",
                    "content": [{"text": "command parameter required for send action"}]
                }
            
            if not _clients:
                return {
                    "status": "error",
                    "content": [{"text": "No browser connected. Load extension first."}]
                }
            
            # Send command and wait for response
            response = send_command_sync(command, params or {}, timeout=30)
            
            # Check for error
            if 'error' in response:
                return {
                    "status": "error",
                    "content": [{"text": f"Browser error: {response['error']}"}]
                }
            
            # Handle screenshot special case - return as image
            if command == "screenshot" and response.get('success') and response.get('screenshot'):
                # Convert base64 to bytes
                screenshot_data = response['screenshot']
                # Remove data:image/png;base64, prefix if present
                if ',' in screenshot_data:
                    screenshot_data = screenshot_data.split(',', 1)[1]
                
                image_bytes = base64.b64decode(screenshot_data)
                
                return {
                    "status": "success",
                    "content": [{
                        "image": {
                            "format": "png",
                            "source": {"bytes": image_bytes}
                        }
                    }]
                }
            
            # Handle other commands - return as text
            return {
                "status": "success",
                "content": [{"text": f"✓ Result: {json.dumps(response, indent=2)}"}]
            }
        
        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}. Use: start_server, stop_server, status, send"}]
            }
    
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}]
        }
