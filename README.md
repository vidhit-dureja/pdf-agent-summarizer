# PDF Agent Summarizer

Tiny Python agent that reads all PDFs in a folder and generates structured summaries using the OpenAI API.

---

## What it does

- Scans an **input folder** for `.pdf` files  
- Extracts text using `pypdf`  
- Splits long documents into overlapping **chunks**  
- Summarizes each chunk with an OpenAI model  
- Combines chunk summaries into a clean, structured **Markdown summary**  
- Writes one `.summary.md` file per PDF into an **output folder**  
- Skips PDFs that already have a summary (unless you use `--force`)  

---

## Features

- 🔧 Command-line flags (`argparse`)
- 📁 Configurable input & output folders
- 🧠 Pluggable OpenAI model (default: `gpt-4.1-mini`)
- 📝 Multiple summary “styles” (default, bullet, narrative, executive)
- 📚 Chunking support for long PDFs
- ✅ Idempotent: skips already summarized files unless `--force`
- 💥 Graceful error handling (bad PDFs, missing API key, API errors)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/vidhit-dureja/pdf-agent-summarizer.git
cd pdf-agent-summarizer
