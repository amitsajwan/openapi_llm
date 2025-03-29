document.addEventListener("DOMContentLoaded", function () {
    const submitBtn = document.getElementById("submit");
    const urlInput = document.getElementById("url_input");
    const fileInput = document.getElementById("file_input");

    submitBtn.addEventListener("click", async function (event) {
        event.preventDefault();  // Prevent form submission

        // Get the URL or File input
        const url = urlInput.value.trim();
        const file = fileInput.files[0];

        if (url || file) {
            const formData = new FormData();
            if (file) {
                formData.append("file", file);
            }
            if (url) {
                formData.append("url", url);
            }

            try {
                const response = await fetch("/load_openapi", {
                    method: "POST",
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log(data);
                    if (data.error) {
                        alert(data.error);
                    } else {
                        alert("OpenAPI data loaded successfully!");
                    }
                } else {
                    alert("Failed to load OpenAPI data. Please try again.");
                }
            } catch (error) {
                console.error("Error:", error);
                alert("An error occurred while processing the request.");
            }
        } else {
            alert("Please provide a Swagger URL or file.");
        }
    });
});
