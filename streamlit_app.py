import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    return tokenizer, model


def generate_response(message, history, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful educational science chatbot.",
        },
        *history[-4:],
        {"role": "user", "content": message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=True
    ).strip()


st.set_page_config(page_title="Educational Chatbot", page_icon="🎓")
st.title("Educational Chatbot")
st.caption(
    "Ask science, physics, or AI questions. Powered by SmolLM2-360M-Instruct."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask an educational question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            tokenizer, model = load_model()
            response = generate_response(
                prompt, st.session_state.messages[:-1], tokenizer, model
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
