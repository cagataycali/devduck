/**
 * 🦆 DevDuck Bridge — Content Script
 *
 * Injected into every page. Handles:
 * - Element highlighting for visual feedback
 * - Click coordinate passthrough
 * - DOM observation
 */

// Listen for messages from background script
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "highlight") {
    highlightElement(msg.selector);
    sendResponse({ status: "ok" });
  } else if (msg.action === "get_elements_at") {
    const el = document.elementFromPoint(msg.x, msg.y);
    sendResponse({
      tag: el?.tagName,
      id: el?.id,
      className: el?.className,
      text: el?.textContent?.substring(0, 200),
      href: el?.href,
      rect: el?.getBoundingClientRect(),
    });
  } else if (msg.action === "get_clickable") {
    // Find all clickable elements with their bounding boxes
    const clickable = [];
    const selectors = "a, button, input, select, textarea, [onclick], [role='button'], [tabindex]";
    document.querySelectorAll(selectors).forEach((el, i) => {
      if (i > 200) return; // Limit
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        clickable.push({
          tag: el.tagName,
          text: (el.textContent || el.value || el.placeholder || "").substring(0, 80).trim(),
          href: el.href || null,
          type: el.type || null,
          x: Math.round(rect.x + rect.width / 2),
          y: Math.round(rect.y + rect.height / 2),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
        });
      }
    });
    sendResponse({ clickable });
  }
  return true;
});

function highlightElement(selector) {
  // Remove existing highlights
  document.querySelectorAll(".devduck-highlight").forEach(el => el.remove());

  if (!selector) return;

  const target = document.querySelector(selector);
  if (!target) return;

  const rect = target.getBoundingClientRect();
  const overlay = document.createElement("div");
  overlay.className = "devduck-highlight";
  overlay.style.cssText = `
    position: fixed;
    left: ${rect.left}px;
    top: ${rect.top}px;
    width: ${rect.width}px;
    height: ${rect.height}px;
    border: 2px solid #61afef;
    background: rgba(97, 175, 239, 0.1);
    pointer-events: none;
    z-index: 999999;
    transition: all 0.2s;
  `;
  document.body.appendChild(overlay);

  // Auto-remove after 3 seconds
  setTimeout(() => overlay.remove(), 3000);
}

// Notify background that content script is loaded
console.log("🦆 DevDuck Bridge content script loaded");
