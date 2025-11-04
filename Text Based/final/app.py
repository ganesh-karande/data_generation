from flask import Flask, render_template, request, send_file, jsonify
import subprocess
import pandas as pd
import io
import re
import os

app = Flask(__name__)

OUTPUT_FILE = "generated_dataset.xlsx"

def generate_with_llama2(prompt):
    """Run LLaMA 2 locally via Ollama CLI"""
    result = subprocess.run(
        [
            "ollama", "run", "llama2",
            f"Generate a dataset ONLY in valid CSV format (no markdown, no extra text). "
            f"Include headers and data rows. For example:\n"
            f"Name,Age,Grade\nAlice,20,A\nBob,22,B\n\nPrompt: {prompt}"
        ],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Step 1: Ask LLaMA to generate CSV-like text
    model_output = generate_with_llama2(prompt)

    # Step 2: Clean output – remove markdown/code fences or text before headers
    clean_output = model_output

    # Remove markdown fences like ```csv or ```text
    clean_output = re.sub(r"```.*?```", "", clean_output, flags=re.S)
    clean_output = clean_output.replace("```csv", "").replace("```", "")

    # Split into lines and keep only CSV portion
    lines = [line for line in clean_output.strip().splitlines() if line.strip()]
    csv_start_index = next((i for i, line in enumerate(lines) if "," in line), 0)
    csv_text = "\n".join(lines[csv_start_index:])

    # Step 3: Try to parse as CSV
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        print("DEBUG - raw model output:\n", model_output)
        print("DEBUG - cleaned CSV text:\n", csv_text)
        return jsonify({
            "error": "Model did not output valid CSV",
            "details": str(e)
        }), 400

    # Step 4: Save to Excel
    df.to_excel(OUTPUT_FILE, index=False)

    return send_file(OUTPUT_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
