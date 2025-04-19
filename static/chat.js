const socket = new WebSocket("ws://localhost:8000/ws");

socket.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "payload_confirmation") {
    renderPayloadConfirmation(msg);
  } else if (msg.type === "api_response") {
    addChatMessage("api", JSON.stringify(msg.result, null, 2));
  }
};

function renderPayloadConfirmation(msg) {
  const container = document.getElementById("chat-container");

  // Create prompt message
  const prompt = document.createElement("div");
  prompt.textContent = msg.prompt || "Please confirm the payload:";
  container.appendChild(prompt);

  // Create textarea with payload
  const textarea = document.createElement("textarea");
  textarea.id = "payload-editor";
  textarea.rows = 10;
  textarea.cols = 50;
  textarea.value = JSON.stringify(msg.payload, null, 2);
  container.appendChild(textarea);

  // Create confirm button
  const confirmButton = document.createElement("button");
  confirmButton.textContent = "Confirm";
  confirmButton.onclick = () => {
    try {
      const editedPayload = JSON.parse(textarea.value);
      socket.send(JSON.stringify({
        type: "user_payload_confirmation",
        payload: editedPayload
      }));
      addChatMessage("human", "✅ Payload confirmed");
    } catch (e) {
      alert("Invalid JSON. Please correct the payload.");
    }
  };
  container.appendChild(confirmButton);
}

function addChatMessage(sender, message) {
  const container = document.getElementById("chat-container");
  const msgDiv = document.createElement("div");
  msgDiv.className = sender;
  msgDiv.textContent = message;
  container.appendChild(msgDiv);
}
