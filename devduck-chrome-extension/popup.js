// 🦆 DevDuck Bridge Popup

const resultEl = document.getElementById("result");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

function showResult(data) {
  resultEl.style.display = "block";
  resultEl.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

// Check connection status
chrome.runtime.sendMessage({ action: "ping" }, (response) => {
  if (response?.status === "pong") {
    statusDot.classList.add("connected");
    statusText.textContent = "Connected to DevDuck";
  } else {
    statusText.textContent = "Disconnected";
  }
});

document.getElementById("btnScreenshot").addEventListener("click", async () => {
  const response = await chrome.runtime.sendMessage({ action: "screenshot" });
  if (response?.status === "success") {
    showResult(`Screenshot captured (${response.title})\nSize: ${response.data?.length} chars`);
  } else {
    showResult(response?.message || "Error");
  }
});

document.getElementById("btnInfo").addEventListener("click", async () => {
  const response = await chrome.runtime.sendMessage({ action: "info" });
  showResult(response);
});

document.getElementById("btnTabs").addEventListener("click", async () => {
  const response = await chrome.runtime.sendMessage({ action: "tabs" });
  if (response?.tabs) {
    showResult(response.tabs.map(t => `${t.active ? "→ " : "  "}[${t.id}] ${t.title}`).join("\n"));
  }
});

document.getElementById("btnClickable").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const response = await chrome.tabs.sendMessage(tab.id, { action: "get_clickable" });
  if (response?.clickable) {
    showResult(response.clickable.map(c =>
      `${c.tag} "${c.text}" at (${c.x},${c.y})`
    ).join("\n"));
  }
});
