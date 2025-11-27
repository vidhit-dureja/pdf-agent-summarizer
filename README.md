# PDF Agent Summarizer

Tiny Python agent that reads all PDFs in the `docs/` folder and generates structured summaries using the OpenAI API.

## What it does

- Scans the `docs/` folder for `.pdf` files  
- Extracts text using `pypdf`  
- Sends the text to OpenAI (`gpt-4.1-mini`)  
- Saves a summary as `<filename>.summary.txt`  
- Skips PDFs that already have a summary file  

## Setup

Clone the repo:

```bash
git clone https://github.com/vidhit-dureja/pdf-agent-summarizer.git
cd pdf-agent-summarizer
