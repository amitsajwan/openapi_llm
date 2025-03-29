document.getElementById('openapiForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const url = document.getElementById('url').value;
    const fileInput = document.getElementById('file');

    const formData = new FormData();
    if (url) {
        formData.append('url', url);
    }
    if (fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    }

    const messageElement = document.getElementById('message');

    try {
        const response = await fetch('/submit_openapi', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to submit OpenAPI data');
        }

        const data = await response.json();
        messageElement.textContent = data.message || 'OpenAPI data loaded successfully!';
        messageElement.style.color = 'green';
    } catch (error) {
        messageElement.textContent = `Error: ${error.message}`;
        messageElement.style.color = 'red';
    }
});
