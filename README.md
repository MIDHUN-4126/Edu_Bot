# Educational Chatbot

A Gradio- and Streamlit-based educational science chatbot powered by a
Hugging Face instruction-tuned language model.

## Features

- Answers science, physics, and AI questions.
- Uses the last few conversation turns as context.
- Runs on CUDA when available and falls back to CPU.
- Provides a browser-based Gradio chat interface.

## Requirements

- Python 3.10 or later
- Internet access on the first run to download the TinyLlama model
- Optional NVIDIA CUDA setup for GPU acceleration

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open the local Gradio URL shown in the terminal, usually
`http://127.0.0.1:7860`.

To run the Streamlit version:

```powershell
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/).
3. Select `MIDHUN-4126/Edu_Bot`.
4. Set the branch to `main` and the main file to `streamlit_app.py`.
5. Click **Deploy**.

The first deployment downloads the model and may take several minutes. The
Streamlit deployment uses the smaller
`HuggingFaceTB/SmolLM2-360M-Instruct` model to fit Streamlit Community
Cloud's free memory limits. The original Gradio app continues to use
TinyLlama.

The first launch downloads the model from Hugging Face and may take some
time. The model is cached locally for subsequent launches.

## Project files

- `app.py` - chatbot application and Gradio interface
- `streamlit_app.py` - Streamlit deployment entry point
- `requirements.txt` - Python dependencies
