<script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.0.2/mermaid.min.js"></script>
<script>
    mermaid.initialize({ startOnLoad: true });

    const socket = new WebSocket("ws://localhost:8000/ws");
    socket.onmessage = (event) => {
        document.getElementById("mermaid-diagram").innerHTML = '```mermaid\n' + event.data + '\n```';
        mermaid.init(undefined, ".mermaid");
    };
</script>
