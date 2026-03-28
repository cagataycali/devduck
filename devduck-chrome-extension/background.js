/**
 * 🦆 DevDuck Bridge — Chrome Extension Background Service Worker
 *
 * Connects to DevDuck via:
 * 1. Native Messaging (stdin/stdout to devduck process)
 * 2. WebSocket (to devduck mesh relay on ws://localhost:10000)
 *
 * Capabilities:
 * - Screenshot any tab
 * - Navigate, click, type, scroll
 * - Read DOM / execute JS
 * - List & switch tabs
 * - Access cookies and logged-in sessions
 * - Forward CDP commands to any tab via chrome.debugger
 */

const NATIVE_HOST = "com.devduck.bridge";
const WS_URL = "ws://localhost:10000";
const RECONNECT_INTERVAL = 5000;

let wsConnection = null;
let nativePort = null;
let connectedTabId = null;
let debuggerAttached = new Set();

// ═══════════════════════════════════════════════════════════════
// WebSocket Connection to DevDuck Mesh
// ═══════════════════════════════════════════════════════════════

function connectWebSocket() {
  try {
    wsConnection = new WebSocket(WS_URL);

    wsConnection.onopen = () => {
      console.log("🦆 Connected to DevDuck mesh");
      // Register as browser agent
      wsConnection.send(JSON.stringify({
        type: "register",
        agent_id: "chrome-extension",
        agent_type: "browser",
        capabilities: ["screenshot", "navigate", "click", "type", "scroll", "tabs", "dom", "eval", "cookies"],
      }));
      updateBadge("ON", "#4CAF50");
    };

    wsConnection.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        const response = await handleCommand(msg);
        if (response) {
          wsConnection.send(JSON.stringify({
            type: "response",
            request_id: msg.request_id || msg.id,
            ...response,
          }));
        }
      } catch (e) {
        console.error("🦆 Message handling error:", e);
      }
    };

    wsConnection.onclose = () => {
      console.log("🦆 Disconnected from DevDuck mesh");
      updateBadge("OFF", "#F44336");
      setTimeout(connectWebSocket, RECONNECT_INTERVAL);
    };

    wsConnection.onerror = (err) => {
      console.error("🦆 WebSocket error:", err);
    };
  } catch (e) {
    console.error("🦆 WebSocket connection failed:", e);
    setTimeout(connectWebSocket, RECONNECT_INTERVAL);
  }
}

// ═══════════════════════════════════════════════════════════════
// Native Messaging Connection
// ═══════════════════════════════════════════════════════════════

function connectNative() {
  try {
    nativePort = chrome.runtime.connectNative(NATIVE_HOST);

    nativePort.onMessage.addListener(async (msg) => {
      const response = await handleCommand(msg);
      if (response) {
        nativePort.postMessage(response);
      }
    });

    nativePort.onDisconnect.addListener(() => {
      console.log("🦆 Native messaging disconnected:", chrome.runtime.lastError?.message);
      nativePort = null;
    });

    console.log("🦆 Native messaging connected");
  } catch (e) {
    console.log("🦆 Native messaging not available (this is normal if host not installed)");
  }
}

// ═══════════════════════════════════════════════════════════════
// Command Handler
// ═══════════════════════════════════════════════════════════════

async function handleCommand(msg) {
  const action = msg.action || msg.type;

  try {
    switch (action) {
      case "screenshot":
        return await captureScreenshot(msg.tabId);

      case "navigate":
        return await navigateTab(msg.url, msg.tabId);

      case "click":
        return await clickElement(msg.x, msg.y, msg.tabId, msg.selector);

      case "type":
        return await typeText(msg.text, msg.tabId, msg.selector);

      case "scroll":
        return await scrollPage(msg.direction, msg.amount, msg.tabId);

      case "tabs":
        return await listTabs();

      case "switch_tab":
        return await switchTab(msg.tabId);

      case "dom":
        return await getDOM(msg.selector, msg.tabId);

      case "eval":
        return await evaluateJS(msg.expression, msg.tabId);

      case "info":
        return await getPageInfo(msg.tabId);

      case "cookies":
        return await getCookies(msg.url);

      case "cdp":
        return await sendCDP(msg.method, msg.params, msg.tabId);

      case "ping":
        return { status: "pong", timestamp: Date.now() };

      default:
        return { status: "error", message: `Unknown action: ${action}` };
    }
  } catch (e) {
    return { status: "error", message: e.message };
  }
}

// ═══════════════════════════════════════════════════════════════
// Actions
// ═══════════════════════════════════════════════════════════════

async function captureScreenshot(tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  // Use chrome.tabs.captureVisibleTab for simple screenshots
  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
    format: "png",
    quality: 90,
  });

  return {
    status: "success",
    action: "screenshot",
    data: dataUrl,  // data:image/png;base64,...
    tabId: tab.id,
    title: tab.title,
    url: tab.url,
    width: tab.width,
    height: tab.height,
  };
}

async function navigateTab(url, tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  await chrome.tabs.update(tab.id, { url });

  // Wait for navigation
  await new Promise((resolve) => {
    const listener = (updatedTabId, changeInfo) => {
      if (updatedTabId === tab.id && changeInfo.status === "complete") {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    };
    chrome.tabs.onUpdated.addListener(listener);
    setTimeout(resolve, 10000); // Timeout
  });

  return { status: "success", action: "navigate", url, tabId: tab.id };
}

async function clickElement(x, y, tabId, selector) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  if (selector) {
    // Click by CSS selector
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (sel) => {
        const el = document.querySelector(sel);
        if (el) {
          el.click();
          return { clicked: true, tag: el.tagName, text: el.textContent?.substring(0, 100) };
        }
        return { clicked: false };
      },
      args: [selector],
    });
    return { status: "success", action: "click", ...results[0]?.result };
  }

  if (x !== undefined && y !== undefined) {
    // Click by coordinates using debugger
    await ensureDebugger(tab.id);
    await chrome.debugger.sendCommand({ tabId: tab.id }, "Input.dispatchMouseEvent", {
      type: "mousePressed", x, y, button: "left", clickCount: 1,
    });
    await chrome.debugger.sendCommand({ tabId: tab.id }, "Input.dispatchMouseEvent", {
      type: "mouseReleased", x, y, button: "left", clickCount: 1,
    });
    return { status: "success", action: "click", x, y };
  }

  return { status: "error", message: "Provide x,y coordinates or selector" };
}

async function typeText(text, tabId, selector) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  if (selector) {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (sel, txt) => {
        const el = document.querySelector(sel);
        if (el) {
          el.focus();
          el.value = txt;
          el.dispatchEvent(new Event("input", { bubbles: true }));
          el.dispatchEvent(new Event("change", { bubbles: true }));
        }
      },
      args: [selector, text],
    });
    return { status: "success", action: "type", text, selector };
  }

  // Type character by character via debugger
  await ensureDebugger(tab.id);
  for (const char of text) {
    await chrome.debugger.sendCommand({ tabId: tab.id }, "Input.dispatchKeyEvent", {
      type: "keyDown", text: char, key: char, unmodifiedText: char,
    });
    await chrome.debugger.sendCommand({ tabId: tab.id }, "Input.dispatchKeyEvent", {
      type: "keyUp", key: char,
    });
  }
  return { status: "success", action: "type", text };
}

async function scrollPage(direction = "down", amount = 300, tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  const scrollMap = {
    up: [0, -amount],
    down: [0, amount],
    left: [-amount, 0],
    right: [amount, 0],
  };
  const [dx, dy] = scrollMap[direction] || [0, amount];

  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (x, y) => window.scrollBy(x, y),
    args: [dx, dy],
  });

  return { status: "success", action: "scroll", direction, amount };
}

async function listTabs() {
  const tabs = await chrome.tabs.query({});
  return {
    status: "success",
    action: "tabs",
    tabs: tabs.map(t => ({
      id: t.id,
      title: t.title,
      url: t.url,
      active: t.active,
      windowId: t.windowId,
      favIconUrl: t.favIconUrl,
    })),
  };
}

async function switchTab(tabId) {
  await chrome.tabs.update(tabId, { active: true });
  const tab = await chrome.tabs.get(tabId);
  await chrome.windows.update(tab.windowId, { focused: true });
  return { status: "success", action: "switch_tab", tabId, title: tab.title, url: tab.url };
}

async function getDOM(selector, tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (sel) => {
      if (sel) {
        const el = document.querySelector(sel);
        return el ? el.outerHTML : null;
      }
      return document.documentElement.outerHTML.substring(0, 100000);
    },
    args: [selector || null],
  });

  return {
    status: "success",
    action: "dom",
    html: results[0]?.result,
    selector,
  };
}

async function evaluateJS(expression, tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (expr) => {
      try {
        return { value: eval(expr), error: null };
      } catch (e) {
        return { value: null, error: e.message };
      }
    },
    args: [expression],
  });

  const result = results[0]?.result;
  if (result?.error) {
    return { status: "error", action: "eval", message: result.error };
  }
  return { status: "success", action: "eval", value: result?.value };
}

async function getPageInfo(tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => ({
      title: document.title,
      url: window.location.href,
      scrollY: window.scrollY,
      scrollHeight: document.body.scrollHeight,
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth,
      readyState: document.readyState,
    }),
  });

  return { status: "success", action: "info", ...results[0]?.result, tabId: tab.id };
}

async function getCookies(url) {
  const cookies = url
    ? await chrome.cookies.getAll({ url })
    : await chrome.cookies.getAll({});

  return {
    status: "success",
    action: "cookies",
    count: cookies.length,
    cookies: cookies.map(c => ({
      name: c.name,
      domain: c.domain,
      path: c.path,
      secure: c.secure,
      httpOnly: c.httpOnly,
      // Don't send values for security — only names
      hasValue: !!c.value,
    })),
  };
}

async function sendCDP(method, params, tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : (await getActiveTab());
  if (!tab) return { status: "error", message: "No active tab" };

  await ensureDebugger(tab.id);
  const result = await chrome.debugger.sendCommand({ tabId: tab.id }, method, params || {});
  return { status: "success", action: "cdp", method, result };
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function ensureDebugger(tabId) {
  if (!debuggerAttached.has(tabId)) {
    await chrome.debugger.attach({ tabId }, "1.3");
    debuggerAttached.add(tabId);

    // Clean up on detach
    chrome.debugger.onDetach.addListener((source) => {
      debuggerAttached.delete(source.tabId);
    });
  }
}

function updateBadge(text, color) {
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
}

// ═══════════════════════════════════════════════════════════════
// Message handler for popup and content scripts
// ═══════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  handleCommand(msg).then(sendResponse);
  return true; // async
});

// ═══════════════════════════════════════════════════════════════
// Initialize
// ═══════════════════════════════════════════════════════════════

connectWebSocket();
connectNative();
console.log("🦆 DevDuck Bridge extension loaded");
