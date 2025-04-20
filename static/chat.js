const socket = new WebSocket("ws://localhost:8000/ws");

// Append a chat bubble to #chat-messages
function addChatMessage(type, text) {
  const container = document.getElementById("chat-messages");
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.textContent = text;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

socket.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "payload_confirmation") {
    renderPayloadConfirmation(msg);
  } else if (msg.type === "api_response") {
    addChatMessage("api", JSON.stringify(msg.result, null, 2));
  } else if (msg.type === "text") {
    addChatMessage("api", msg.text || msg.message || JSON.stringify(msg));
  }
};

function renderPayloadConfirmation(msg) {
  // Create one bubble for payload confirmation
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message payload";

  // Prompt
  const promptEl = document.createElement("div");
  promptEl.textContent = msg.prompt || "Please confirm the payload:";
  bubble.appendChild(promptEl);

  // Textarea for editing JSON
  const textarea = document.createElement("textarea");
  textarea.id = "payload-editor";
  textarea.value = JSON.stringify(msg.payload, null, 2);
  bubble.appendChild(textarea);

  // Confirm button
  const confirmBtn = document.createElement("button");
  confirmBtn.textContent = "✅ Confirm Payload";
  confirmBtn.onclick = () => {
    let edited;
    try {
      edited = JSON.parse(textarea.value);
    } catch (e) {
      alert("Invalid JSON. Please correct the payload.");
      return;
    }

    // Send edited payload back to server
    socket.send(JSON.stringify({
      type: "user_payload_confirmation",
      payload: edited
    }));
    // Show user confirmation bubble
    addChatMessage("human", "✅ Payload confirmed");
  };
  bubble.appendChild(confirmBtn);

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

// Send button logic
document.getElementById("send-button").onclick = () => {
  const inp = document.getElementById("user-input");
  const text = inp.value.trim();
  if (!text) return;
  socket.send(JSON.stringify({ type: "user_message", text }));
  addChatMessage("human", text);
  inp.value = "";
};
