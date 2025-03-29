const socket = new WebSocket("ws://localhost:8000/chat");

socket.onmessage = function(event) {
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<div class="message bot">${event.data}</div>`;
};

function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (message !== "") {
        socket.send(message);
        document.getElementById("chat-box").innerHTML += `<div class="message user">${message}</div>`;
        input.value = "";
    }
}
