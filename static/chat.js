const socket = new WebSocket("ws://localhost:8000/ws");

socket.onmessage = function (event) {
    const msg = JSON.parse(event.data);
    renderMessage(
        msg.type || msg.sender || "bot",
        msg.message || msg.response || JSON.stringify(msg, null, 2)
    );
};

function renderMessage(type, content) {
    const chatBox = document.getElementById("chatBox");
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${type}`;

    const timestamp = new Date().toLocaleTimeString();

    const timestampDiv = document.createElement("div");
    timestampDiv.className = "timestamp";
    timestampDiv.innerText = timestamp;
    msgDiv.appendChild(timestampDiv);

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    if (isJson(content)) {
        contentDiv.innerHTML = `<pre><code class="json">${syntaxHighlight(content)}</code></pre>`;
        hljs.highlightElement(contentDiv.querySelector("code"));
    } else if (content.includes("```") || content.includes("#")) {
        contentDiv.innerHTML = marked.parse(content);
    } else {
        contentDiv.innerText = content;
    }

    msgDiv.appendChild(contentDiv);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById("userInput");
    const text = input.value.trim();
    if (text) {
        renderMessage("human", text);
        socket.send(JSON.stringify({ message: text }));
        input.value = "";
    }
}

function handleEnter(event) {
    if (event.key === "Enter") sendMessage();
}

function uploadOpenAPI() {
    const file = document.getElementById("openapiFile").files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => socket.send(JSON.stringify({ message: reader.result }));
        reader.readAsText(file);
    }
}

function submitOpenAPIUrl() {
    const url = document.getElementById("openapiUrl").value.trim();
    if (url) {
        socket.send(JSON.stringify({ message: url }));
    }
}

function isJson(text) {
    try {
        JSON.parse(text);
        return true;
    } catch {
        return false;
    }
}

function syntaxHighlight(json) {
    if (typeof json !== "string") {
        json = JSON.stringify(json, null, 2);
    }
    return json.replace(/&/g, "&amp;")
               .replace(/</g, "&lt;")
               .replace(/>/g, "&gt;");
}
