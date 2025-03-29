const websocket = new WebSocket("ws://localhost:8000/chat");

websocket.onmessage = function(event) {
    document.getElementById("chat-log").value += `System: ${event.data}\n`;
    document.getElementById("chat-log").scrollTop = document.getElementById("chat-log").scrollHeight;
};

function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (userInput) {
        websocket.send(userInput);
        document.getElementById("chat-log").value += `You: ${userInput}\n`;
        document.getElementById("user-input").value = '';
    }
}

function submitOpenAPI() {
    const swaggerUrl = document.getElementById("swagger-url").value;
    const fileInput = document.getElementById("swagger-file");

    if (swaggerUrl || fileInput.files.length > 0) {
        const formData = new FormData();
        if (swaggerUrl) {
            formData.append('url_or_file', swaggerUrl);
        } else {
            formData.append('url_or_file', fileInput.files[0]);
        }

        fetch('/load_openapi/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                document.getElementById("chat-log").value += "OpenAPI loaded successfully.\n";
            } else {
                document.getElementById("chat-log").value += "Failed to load OpenAPI.\n";
            }
        });
    } else {
        alert("Please enter a URL or upload a file.");
    }
}
