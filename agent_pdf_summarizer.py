import os
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

client = OpenAI()

# ---------- 1. "Perception": read PDF text ----------
def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n\n".join(chunks)

# ---------- 2. "Thinking": ask LLM for summary ----------
def summarize_text(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise study assistant that summarizes documents for a college student."
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following document into:\n"
                    "1) A 5–7 sentence overview\n"
                    "2) 3–5 bullet key ideas\n"
                    "3) 3 possible exam questions.\n\n"
                    f"Document:\n{text[:15000]}"
                )
            }
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# ---------- 3. "Action": save summary to .txt ----------
def save_summary(pdf_path: Path, summary: str) -> Path:
    output_path = pdf_path.with_suffix(".summary.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return output_path

# ---------- 4. Agent loop over a folder ----------
def run_agent_on_folder(folder: str = "docs"):
    docs_dir = Path(folder)
    docs_dir.mkdir(exist_ok=True)

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {docs_dir.resolve()}")
        return

    for pdf in pdf_files:
        summary_path = pdf.with_suffix(".summary.txt")
        if summary_path.exists():
            print(f"Skipping {pdf.name} (summary already exists).")
            continue

        print(f"\n=== Processing {pdf.name} ===")
        text = read_pdf_text(pdf)
        print(f"Read {len(text)} characters from PDF.")

        summary = summarize_text(text)
        out_path = save_summary(pdf, summary)
        print(f"Summary saved to: {out_path.name}")

if __name__ == "__main__":
    run_agent_on_folder("docs")

