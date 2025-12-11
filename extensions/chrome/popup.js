// DevDuck Chat Extension
const msgs = document.getElementById('messages');
const input = document.getElementById('messageInput');
const btn = document.getElementById('sendBtn');
const title = document.getElementById('title');
const form = document.getElementById('messageForm');

// Configure marked for markdown parsing
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true
});

// Turn tracking - maps turn_id to DOM elements
const turns = new Map();

function createTurn(turnId, userMessage) {
    const turnDiv = document.createElement('div');
    turnDiv.className = 'turn';
    turnDiv.dataset.turnId = turnId;

    // User message
    const userDiv = document.createElement('div');
    userDiv.className = 'message user';
    userDiv.textContent = userMessage;
    turnDiv.appendChild(userDiv);

    // Assistant message container
    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'message assistant';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    assistantDiv.appendChild(content);

    // Loading cursor
    const cursor = document.createElement('span');
    cursor.className = 'loading-cursor';
    assistantDiv.appendChild(cursor);

    turnDiv.appendChild(assistantDiv);

    msgs.appendChild(turnDiv);
    msgs.scrollTop = msgs.scrollHeight;

    turns.set(turnId, {
        turnDiv,
        assistantDiv,
        content,
        cursor,
        text: '',
        tools: []
    });

    return turns.get(turnId);
}

function updateTurnContent(turnId, newText) {
    const turn = turns.get(turnId);
    if (!turn) return;

    turn.text += newText;

    // Parse and render markdown
    const html = marked.parse(turn.text);
    turn.content.innerHTML = html;

    // Highlight code blocks
    turn.content.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });

    msgs.scrollTop = msgs.scrollHeight;
}

function addToolStatus(turnId, toolName, toolNumber) {
    const turn = turns.get(turnId);
    if (!turn) return;

    const toolDiv = document.createElement('div');
    toolDiv.className = 'tool-status';
    toolDiv.innerHTML = `🛠️ #${toolNumber}: ${toolName}`;
    
    // Insert before content
    turn.assistantDiv.insertBefore(toolDiv, turn.content);
    turn.tools.push(toolDiv);

    msgs.scrollTop = msgs.scrollHeight;
}

function updateToolStatus(turnId, success) {
    const turn = turns.get(turnId);
    if (!turn || turn.tools.length === 0) return;

    const lastTool = turn.tools[turn.tools.length - 1];
    lastTool.className = `tool-status ${success ? 'success' : 'error'}`;
    lastTool.innerHTML += success ? ' ✅' : ' ❌';
}

function finalizeTurn(turnId) {
    const turn = turns.get(turnId);
    if (!turn) return;

    // Remove loading cursor
    if (turn.cursor && turn.cursor.parentNode) {
        turn.cursor.remove();
    }
}

// Auto-resize textarea
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 96) + 'px';
});

let ws = null;

function connect() {
    try {
        // Connect to DevDuck WebSocket server on port 8080
        ws = new WebSocket('ws://localhost:8080');
        
        ws.onopen = () => {
            title.className = 'connected';
            input.disabled = false;
            btn.disabled = false;
            input.focus();
        };
        
        ws.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                
                switch(msg.type) {
                    case 'connected':
                        // Welcome message received
                        break;
                        
                    case 'turn_start':
                        createTurn(msg.turn_id, msg.data);
                        break;
                        
                    case 'chunk':
                        updateTurnContent(msg.turn_id, msg.data);
                        break;
                        
                    case 'tool_start':
                        addToolStatus(msg.turn_id, msg.data, msg.tool_number);
                        break;
                        
                    case 'tool_end':
                        updateToolStatus(msg.turn_id, msg.success);
                        break;
                        
                    case 'turn_end':
                        finalizeTurn(msg.turn_id);
                        break;
                        
                    case 'error':
                        console.error('Server error:', msg.data);
                        if (msg.turn_id) {
                            finalizeTurn(msg.turn_id);
                        }
                        break;
                        
                    case 'disconnected':
                        break;
                }
            } catch (err) {
                console.error('Error parsing message:', err, e.data);
            }
        };
        
        ws.onerror = () => {
            title.className = '';
            console.error('Connection error. Make sure DevDuck server is running.');
        };
        
        ws.onclose = () => {
            title.className = '';
            input.disabled = true;
            btn.disabled = true;
            
            // Try to reconnect after 3 seconds
            setTimeout(connect, 3000);
        };
        
    } catch (e) {
        console.error('WebSocket error:', e);
    }
}

form.onsubmit = (e) => {
    e.preventDefault();
    
    if (input.value.trim() && ws && ws.readyState === WebSocket.OPEN) {
        const msg = input.value.trim();
        ws.send(msg);
        input.value = '';
        input.style.height = 'auto';
    }
};

// Handle Enter key (without Shift)
input.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!btn.disabled) {
            form.dispatchEvent(new Event('submit'));
        }
    }
};

// Start connection
connect();
