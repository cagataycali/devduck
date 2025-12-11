// DevDuck Browser Bridge - Background Service Worker
let ws = null;
let reconnectInterval = null;
let isConnected = false;

const WS_URL = 'ws://localhost:9223';

// Connect to DevDuck WebSocket server
function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log('[DevDuck] Already connected');
    return;
  }

  console.log('[DevDuck] Connecting to', WS_URL);
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('[DevDuck] Connected');
    isConnected = true;
    chrome.action.setBadgeText({ text: '✓' });
    chrome.action.setBadgeBackgroundColor({ color: '#00FF00' });
    
    if (reconnectInterval) {
      clearInterval(reconnectInterval);
      reconnectInterval = null;
    }

    // Send hello message with ID
    send({ 
      id: crypto.randomUUID(), 
      type: 'hello', 
      data: { version: '1.0.0' } 
    });
    
    // Notify popup of status change
    chrome.runtime.sendMessage({ action: 'statusUpdate', connected: true }).catch(() => {});
  };

  ws.onclose = () => {
    console.log('[DevDuck] Disconnected');
    isConnected = false;
    chrome.action.setBadgeText({ text: '✗' });
    chrome.action.setBadgeBackgroundColor({ color: '#FF0000' });
    
    // Notify popup of status change
    chrome.runtime.sendMessage({ action: 'statusUpdate', connected: false }).catch(() => {});
    
    // Auto-reconnect
    if (!reconnectInterval) {
      reconnectInterval = setInterval(() => {
        console.log('[DevDuck] Reconnecting...');
        connect();
      }, 3000);
    }
  };

  ws.onerror = (error) => {
    console.error('[DevDuck] WebSocket error:', error);
  };

  ws.onmessage = async (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
      console.log('[DevDuck] Received command:', message.command, 'id:', message.id);
      
      // Only handle command messages
      if (message.command) {
        const response = await handleCommand(message);
        console.log('[DevDuck] Command success:', message.command, 'response preview:', JSON.stringify(response).substring(0, 100));
        send({ id: message.id, type: 'response', data: response });
      }
    } catch (error) {
      console.error('[DevDuck] Error handling message:', message?.command, error);
      const errorResponse = { 
        id: message?.id || 'unknown', 
        type: 'error', 
        error: error.message,
        stack: error.stack
      };
      send(errorResponse);
    }
  };
}

// Send message to DevDuck
function send(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log('[DevDuck] Sending response id:', data.id, 'type:', data.type);
    ws.send(JSON.stringify(data));
  } else {
    console.error('[DevDuck] Cannot send - WebSocket not open, state:', ws?.readyState);
  }
}

// Handle commands from DevDuck
async function handleCommand(message) {
  const { command, params = {} } = message;

  switch (command) {
    case 'navigate':
      return await navigate(params.url, params.tabId);
    
    case 'execute':
      return await executeScript(params.code, params.tabId);
    
    case 'screenshot':
      return await takeScreenshot(params.tabId);
    
    case 'getContent':
      return await getPageContent(params.tabId);
    
    case 'click':
      return await clickElement(params.selector, params.tabId);
    
    case 'fill':
      return await fillInput(params.selector, params.value, params.tabId);
    
    case 'getTabs':
      return await getTabs();
    
    case 'closeTab':
      return await closeTab(params.tabId);
    
    case 'newTab':
      return await newTab(params.url);
    
    case 'selectTab':
      return await selectTab(params.tabId);
    
    case 'reloadTab':
      return await reloadTab(params.tabId);
    
    case 'goBack':
      return await goBack(params.tabId);
    
    case 'goForward':
      return await goForward(params.tabId);
    
    case 'getCurrentUrl':
      return await getCurrentUrl(params.tabId);
    
    case 'waitForSelector':
      return await waitForSelector(params.selector, params.tabId, params.timeout);
    
    case 'getConsoleLogs':
      return await getConsoleLogs(params.tabId);
    
    case 'readClipboard':
      return await readClipboard();
    
    case 'writeClipboard':
      return await writeClipboard(params.text);
    
    case 'getLocation':
      return await getLocation(params.tabId);
    
    case 'listUSB':
      return await listUSBDevices(params.tabId);
    
    case 'showNotification':
      return await showNotification(params.title, params.message, params.icon);
    
    case 'getStorage':
      return await getStorage(params.tabId, params.type);
    
    case 'setStorage':
      return await setStorage(params.tabId, params.type, params.key, params.value);
    
    default:
      throw new Error(`Unknown command: ${command}`);
  }
}

// Get current or specified tab
async function getCurrentTab(tabId) {
  if (tabId) {
    return await chrome.tabs.get(tabId);
  }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    throw new Error('No active tab found');
  }
  return tab;
}

// Navigate to URL
async function navigate(url, tabId) {
  const tab = await getCurrentTab(tabId);
  await chrome.tabs.update(tab.id, { url });
  
  // Wait for page to load
  return new Promise((resolve) => {
    chrome.tabs.onUpdated.addListener(function listener(updatedTabId, info) {
      if (updatedTabId === tab.id && info.status === 'complete') {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve({ success: true, url, tabId: tab.id });
      }
    });
  });
}

// Execute JavaScript
async function executeScript(code, tabId) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (code) => {
      try {
        // Direct eval to execute and return result
        const result = eval(code);
        return result;
      } catch (error) {
        return { __error: error.message };
      }
    },
    args: [code]
  });
  
  const result = results[0].result;
  
  // Check for error
  if (result && result.__error) {
    throw new Error(result.__error);
  }
  
  return { success: true, result: result };
}

// Take screenshot
async function takeScreenshot(tabId) {
  const tab = await getCurrentTab(tabId);
  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' });
  return { success: true, screenshot: dataUrl };
}

// Get page content
async function getPageContent(tabId) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => ({
      title: document.title,
      url: window.location.href,
      html: document.documentElement.outerHTML,
      text: document.body.innerText
    })
  });
  
  return { success: true, content: results[0].result };
}

// Click element
async function clickElement(selector, tabId) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (selector) => {
      const element = document.querySelector(selector);
      if (!element) throw new Error(`Element not found: ${selector}`);
      element.click();
      return true;
    },
    args: [selector]
  });
  
  return { success: true, clicked: results[0].result };
}

// Fill input
async function fillInput(selector, value, tabId) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (selector, value) => {
      const element = document.querySelector(selector);
      if (!element) throw new Error(`Element not found: ${selector}`);
      element.value = value;
      element.dispatchEvent(new Event('input', { bubbles: true }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    },
    args: [selector, value]
  });
  
  return { success: true, filled: results[0].result };
}

// Get all tabs
async function getTabs() {
  const tabs = await chrome.tabs.query({});
  return { 
    success: true, 
    tabs: tabs.map(t => ({ 
      id: t.id, 
      url: t.url, 
      title: t.title, 
      active: t.active,
      windowId: t.windowId,
      index: t.index
    })) 
  };
}

// Close tab
async function closeTab(tabId) {
  await chrome.tabs.remove(tabId);
  return { success: true, closed: tabId };
}

// Create new tab
async function newTab(url) {
  const tab = await chrome.tabs.create({ url: url || 'about:blank' });
  return { success: true, tabId: tab.id, url: tab.url };
}

// Switch to tab
async function selectTab(tabId) {
  const tab = await chrome.tabs.get(tabId);
  await chrome.tabs.update(tabId, { active: true });
  await chrome.windows.update(tab.windowId, { focused: true });
  return { success: true, tabId, active: true };
}

// Reload tab
async function reloadTab(tabId) {
  const tab = await getCurrentTab(tabId);
  await chrome.tabs.reload(tab.id);
  return { success: true, reloaded: tab.id };
}

// Go back
async function goBack(tabId) {
  const tab = await getCurrentTab(tabId);
  await chrome.tabs.goBack(tab.id);
  return { success: true, tabId: tab.id };
}

// Go forward
async function goForward(tabId) {
  const tab = await getCurrentTab(tabId);
  await chrome.tabs.goForward(tab.id);
  return { success: true, tabId: tab.id };
}

// Get current URL
async function getCurrentUrl(tabId) {
  const tab = await getCurrentTab(tabId);
  return { success: true, url: tab.url, title: tab.title, tabId: tab.id };
}

// Wait for selector
async function waitForSelector(selector, tabId, timeout = 10000) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (selector, timeout) => {
      return new Promise((resolve, reject) => {
        const startTime = Date.now();
        const check = () => {
          const element = document.querySelector(selector);
          if (element) {
            resolve(true);
          } else if (Date.now() - startTime > timeout) {
            reject(new Error(`Timeout waiting for selector: ${selector}`));
          } else {
            setTimeout(check, 100);
          }
        };
        check();
      });
    },
    args: [selector, timeout]
  });
  
  return { success: true, found: results[0].result };
}

// Handle messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getStatus') {
    sendResponse({ connected: isConnected });
  }
  return true;
});

// ========== ENHANCED FEATURES ==========

// Get console logs from page
async function getConsoleLogs(tabId) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      // Capture console logs
      const logs = [];
      const originalLog = console.log;
      const originalError = console.error;
      const originalWarn = console.warn;
      
      console.log = function(...args) {
        logs.push({ type: 'log', message: args.join(' '), timestamp: Date.now() });
        originalLog.apply(console, args);
      };
      
      console.error = function(...args) {
        logs.push({ type: 'error', message: args.join(' '), timestamp: Date.now() });
        originalError.apply(console, args);
      };
      
      console.warn = function(...args) {
        logs.push({ type: 'warn', message: args.join(' '), timestamp: Date.now() });
        originalWarn.apply(console, args);
      };
      
      // Return captured logs (this won't work for previous logs)
      return { logs, note: 'Logging started - only captures future logs' };
    }
  });
  
  return { success: true, result: results[0].result };
}

// Read clipboard
async function readClipboard() {
  try {
    // Get active tab and execute in page context
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: async () => {
        return await navigator.clipboard.readText();
      }
    });
    
    return { success: true, text: results[0].result };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Write to clipboard
async function writeClipboard(text) {
  try {
    // Get active tab and execute in page context
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: async (textToWrite) => {
        await navigator.clipboard.writeText(textToWrite);
        return true;
      },
      args: [text]
    });
    
    return { success: true, written: results[0].result };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Get geolocation
async function getLocation(tabId) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      return new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
          (position) => resolve({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy,
            altitude: position.coords.altitude,
            heading: position.coords.heading,
            speed: position.coords.speed
          }),
          (error) => reject(error.message),
          { enableHighAccuracy: true }
        );
      });
    }
  });
  
  return { success: true, location: results[0].result };
}

// List USB devices
async function listUSBDevices(tabId) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: async () => {
      try {
        const devices = await navigator.usb.getDevices();
        return devices.map(device => ({
          productId: device.productId,
          vendorId: device.vendorId,
          productName: device.productName,
          manufacturerName: device.manufacturerName,
          serialNumber: device.serialNumber
        }));
      } catch (error) {
        return { error: error.message };
      }
    }
  });
  
  return { success: true, devices: results[0].result };
}

// Show notification
async function showNotification(title, message, icon) {
  try {
    await chrome.notifications.create({
      type: 'basic',
      iconUrl: icon || 'icon48.png',
      title: title || 'DevDuck',
      message: message || ''
    });
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Get storage (localStorage/sessionStorage)
async function getStorage(tabId, type = 'local') {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (type) => {
      const storage = type === 'session' ? sessionStorage : localStorage;
      const data = {};
      for (let i = 0; i < storage.length; i++) {
        const key = storage.key(i);
        data[key] = storage.getItem(key);
      }
      return data;
    },
    args: [type]
  });
  
  return { success: true, storage: results[0].result, type };
}

// Set storage
async function setStorage(tabId, type = 'local', key, value) {
  const tab = await getCurrentTab(tabId);
  
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (type, key, value) => {
      const storage = type === 'session' ? sessionStorage : localStorage;
      storage.setItem(key, value);
      return true;
    },
    args: [type, key, value]
  });
  
  return { success: true, set: results[0].result };
}

// Auto-connect on startup
connect();
