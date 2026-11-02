import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading conversational model: {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
print("Conversational model loaded successfully!")

def gradio_chat(message, history):
    formatted_context = ""
    for turn in (history or [])[-4:]:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            formatted_context += f"<|user|>{turn[0]}<|assistant|>{turn[1]}"
        elif isinstance(turn, dict):
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                formatted_context += f"<|user|>{content}"
            elif role == "assistant":
                formatted_context += f"<|assistant|>{content}"

    prompt = f"<|system|>You are a helpful educational science chatbot.<|user|>{formatted_context}{message}<|assistant|"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

css = """
.gradio-container {
    max-width: 100% !important;
    height: 95vh !important;
    display: flex !important;
    flex-direction: column !important;
}
#chat-wrapper {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}
#chat-wrapper > div {
    flex-grow: 1 !important;
    height: 100% !important;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("# <center>Transformer-Based LLM Chatbot</center")
    gr.Markdown("<center>Ask science, physics, or AI questions. Powered by TinyLlama-1.1B-Chat.</center>")
    
    with gr.Column(elem_id="chat-wrapper"):
        gr.ChatInterface(
            fn=gradio_chat,
            examples=["What is physics?", "Explain artificial intelligence in simple words."],
            cache_examples=False
        )

demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"), css=css)