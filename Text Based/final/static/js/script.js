document.getElementById("generateBtn").addEventListener("click", async () => {
    const prompt = document.getElementById("prompt").value.trim();
    const loading = document.getElementById("loading");
    const message = document.getElementById("message");

    if (!prompt) {
        alert("Please enter a prompt!");
        return;
    }

    loading.style.display = "block";
    message.innerText = "";

    try {
        const response = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });

        loading.style.display = "none";

        if (!response.ok) {
            const error = await response.json();
            message.innerText = "❌ Error: " + (error.error || "Failed to generate dataset");
            return;
        }

        // Create and trigger download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "generated_dataset.xlsx";
        document.body.appendChild(a);
        a.click();
        a.remove();

        message.innerText = "✅ Dataset generated successfully!";
    } catch (err) {
        loading.style.display = "none";
        message.innerText = "❌ Something went wrong.";
        console.error(err);
    }
});
