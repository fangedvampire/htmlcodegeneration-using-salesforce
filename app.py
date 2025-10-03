import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import textwrap

# --- MODEL SETTINGS ---
MODEL_ID = "Salesforce/codegen-350M-mono"

@st.cache_resource
def load_model():
    st.info("Loading model... This may take a while on first run.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# --- PROMPT TEMPLATE ---
PROMPT_TEMPLATE = textwrap.dedent("""
You are a helpful assistant that writes clean HTML code based on a description.
Output ONLY the HTML document with inline styles, without explanation.
Use semantic HTML5 elements and make it responsive if possible.

Example:
Title: Italian Bistro Menu
Description: Single-page responsive menu for a small Italian bistro with sections: Appetizers, Pizzas, Pastas, Desserts. Include prices and descriptions.
Style: Warm, rustic, mobile-friendly layout.

HTML:
<!doctype html>
<html lang="en">
<head>...</head>
<body>...</body>
</html>

END EXAMPLE.

USER SPEC:
Title: {title}
Description: {description}
Style: {style}

Generate the HTML now.
""")

def generate_html(title, description, style):
    prompt = PROMPT_TEMPLATE.format(title=title, description=description, style=style)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.02,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Try to remove echoed prompt
    if prompt in result:
        return result.split(prompt)[-1].strip()
    return result.strip()


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI HTML Code Generator", layout="wide")

st.title("🖥 AI HTML Code Generator")
st.markdown("Generate HTML code from a description using the Salesforce CodeGen mono model.")

with st.form("generate_form"):
    title = st.text_input("Page Title", "Sample Page")
    description = st.text_area("Description", "A single-page website showing a restaurant menu with sections for appetizers, main course, and desserts.")
    style = st.text_input("Style Notes", "Responsive, clean, modern layout")
    submitted = st.form_submit_button("Generate HTML")

if submitted:
    with st.spinner("Generating HTML..."):
        html_output = generate_html(title, description, style)
    st.success("HTML Generated ✅")
    st.code(html_output, language="html")
    st.markdown("Preview:")
    st.components.v1.html(html_output, height=600, scrolling=True)
