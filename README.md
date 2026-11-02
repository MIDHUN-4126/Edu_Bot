# Educational Chatbot

A Gradio-based educational science chatbot powered by the
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` language model.

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
python -m pip install -r Requirements.txt
```

## Run

```powershell
python app.py
```

Open the local Gradio URL shown in the terminal, usually
`http://127.0.0.1:7860`.

The first launch downloads the model from Hugging Face and may take some
time. The model is cached locally for subsequent launches.

## Project files

- `app.py` - chatbot application and Gradio interface
- `Requirements.txt` - Python dependencies
