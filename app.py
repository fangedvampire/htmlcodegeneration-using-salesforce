"""
app.py

Flask app that generates HTML code using a Salesforce CodeGen "mono" model.

Behavior:
- If the environment variable HUGGINGFACE_API_TOKEN is set, the app will call Hugging Face
  Inference API (recommended for simplicity & small/dev usage).
- Otherwise it will attempt to load the model locally with Hugging Face Transformers
  (requires model weights & adequate hardware).
- Provides a web endpoint /generate (POST) and a simple CLI.

POST /generate
JSON body:
{
  "title": "Simple page title",
  "description": "A short human description of the page you want (e.g. 'restaurant menu for Italian bistro')",
  "style": "brief description of styling or framework (optional, e.g. 'responsive, tailwind-like, semantic HTML')"
}

Response:
{
  "html": "<!doctype html>..."
}

NOTE: Codegen models can produce code but don't guarantee security or accessibility requirements.
Always review generated HTML before using it in production.
"""

import os
import json
import textwrap
from flask import Flask, request, jsonify, render_template_string

HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")  # if present, use inference API
MODEL_ID = os.environ.get("CODEGEN_MODEL_ID", "Salesforce/codegen-350M-mono")  # default model
# If you prefer a bigger/smaller model, set CODEGEN_MODEL_ID env var before running.

app = Flask(__name__)

# --- Prompt template (in-built) ---
# This is deliberately explicit: instructs the model to return only valid HTML markup,
# a small CSS block, and comments about usage. The template includes one example spec.
PROMPT_TEMPLATE = textwrap.dedent("""
You are a helpful assistant that writes production-quality HTML. Output ONLY the final HTML document.
Do NOT include commentary or explanation. The HTML must be valid, accessible, and self-contained
(put critical CSS inside a <style> block). Avoid external links. Use semantic HTML5 elements.
If the user asks for a component, output only that component wrapped in a minimal HTML page.

EXAMPLES:
###
Spec: title="Italian Bistro Menu", description="A single-page responsive menu for a small Italian bistro. Include sections: Appetizers, Pizzas, Pastas, Desserts. Include prices and short descriptions. Keep design warm, rustic, with simple responsive layout."
Output: (HTML document only)
<!doctype html>
<html lang="en">
<head> ... </head>
<body> ... </body>
</html>
###
END EXAMPLES

USER SPEC:
Title: {title}
Description: {description}
Style notes: {style}

Generate the HTML now.
""")


# --- Helper: call Hugging Face Inference API ---
def hf_inference_api_generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.2):
    """
    Uses Hugging Face Inference API text generation for models that support it.
    Requires HUGGINGFACE_API_TOKEN set in environment.
    """
    import requests

    assert HF_API_TOKEN, "HUGGINGFACE_API_TOKEN not set"

    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "repetition_penalty": 1.02,
            # "stop": ["###"]  # optional
        },
        "options": {"use_cache": False}
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Inference API returns list of dicts or dict with 'generated_text' depending on model/inference config
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # some models return plain text
    if isinstance(data, str):
        return data
    # last fallback: try to pull text from first element
    try:
        return data[0].get("generated_text", "")
    except Exception:
        return json.dumps(data)


# --- Helper: local transformers generate (fallback) ---
def local_transformers_generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.2):
    """
    Load model locally via transformers. Will attempt to run on GPU if available.
    Warning: model weights must be downloaded and sufficient memory must be available.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_id = MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.02,
        eos_token_id=tokenizer.eos_token_id,
    )
    outputs = model.generate(**gen_kwargs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The decode will include the prompt; attempt to return only the generated suffix after the prompt
    if result.startswith(prompt):
        return result[len(prompt):].strip()
    return result.strip()


# --- Main generate wrapper ---
def generate_html_from_spec(title: str, description: str, style: str = ""):
    prompt = PROMPT_TEMPLATE.format(title=title.strip(), description=description.strip(), style=style.strip())
    # Use HF Inference API if token present
    if HF_API_TOKEN:
        raw = hf_inference_api_generate(prompt, max_tokens=1024, temperature=0.2)
    else:
        raw = local_transformers_generate(prompt, max_tokens=1024, temperature=0.2)

    # The model should return an HTML document. If the response includes the prompt echoed, remove possible echo.
    # Try to find first occurrence of "<!doctype" or "<html"
    lowered = raw.lower()
    start_idx = None
    for marker in ("<!doctype", "<html"):
        idx = lowered.find(marker)
        if idx != -1:
            start_idx = idx
            break
    if start_idx is not None:
        html = raw[start_idx:]
    else:
        # fallback: return entire output
        html = raw

    return html.strip()


# --- Flask routes ---
@app.route("/")
def index():
    example = {
        "title": "Italian Bistro Menu",
        "description": "A single-page responsive menu for a small Italian bistro. Sections: Appetizers, Pizzas, Pastas, Desserts. Include prices.",
        "style": "warm, rustic, mobile-first layout"
    }
    return render_template_string(
        "<h2>HTML CodeGen Demo</h2>"
        "<p>POST JSON to <code>/generate</code> to receive generated HTML.</p>"
        "<pre>{{example}}</pre>",
        example=json.dumps(example, indent=2)
    )


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    title = payload.get("title", "").strip()
    description = payload.get("description", "").strip()
    style = payload.get("style", "").strip()

    if not description and not title:
        return jsonify({"error": "Provide at least a title or description"}), 400

    try:
        html = generate_html_from_spec(title=title or "Generated Page", description=description or title, style=style)
        return jsonify({"html": html}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- CLI usage ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML using a CodeGen mono model")
    parser.add_argument("--title", type=str, default="Demo Page")
    parser.add_argument("--description", type=str, required=False, default="A demo single-page website")
    parser.add_argument("--style", type=str, default="clean responsive")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # If invoked directly, run the Flask dev server
    print("Starting Flask app. Use POST /generate to get generated HTML.")
    app.run(host=args.host, port=args.port)
 
