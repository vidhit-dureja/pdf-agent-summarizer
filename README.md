# PDF Agent Summarizer

Tiny Python agent that reads PDFs and generates structured summaries using the OpenAI API. Use it from the command line or via a simple Streamlit web UI.

## What it does

- Scans an input folder for `.pdf` files (CLI)
- Extracts text using `pypdf`
- Splits long documents into overlapping chunks
- Summarizes each chunk with an OpenAI chat model
- Combines chunk summaries into a clean, structured Markdown summary
- Writes one `.summary.md` file per PDF into an output folder
- Skips PDFs that already have a summary (unless you use `--force`)
- Provides a Streamlit app to upload one or more PDFs and download summaries

## Features

- 🔧 Command-line flags (`argparse`)
- 📁 Configurable input & output folders
- 🧠 Pluggable OpenAI model (default: `gpt-4.1-mini`)
- 📝 Multiple summary styles (`default`, `bullet`, `narrative`, `executive`)
- 📚 Chunking support for long PDFs
- ✅ Idempotent: skips already summarized files unless `--force`
- 💥 Graceful error handling (bad PDFs, missing API key, API errors)
- 🌐 Simple Streamlit UI for multi-PDF upload and download

## Setup

1. Clone the repo

   `git clone https://github.com/vidhit-dureja/pdf-agent-summarizer.git`  
   `cd pdf-agent-summarizer`

2. Install dependencies

   `pip install -r requirements.txt`

3. Set your OpenAI API key

   `export OPENAI_API_KEY="sk-..."`

## Usage (CLI)

Summarize all PDFs in a folder:

`python agent_pdf_summarizer.py --input-folder docs --output-folder summaries --model gpt-4.1-mini --style default --max-chars-per-chunk 6000`

Force regenerate all summaries:

`python agent_pdf_summarizer.py --input-folder docs --output-folder summaries --force`

## Usage (Streamlit UI)

Run the web app locally:

`streamlit run app.py`

Then open `http://localhost:8501`, upload one or more PDFs, choose summary style/model, click “Summarize”, and download the `.md` summaries.
