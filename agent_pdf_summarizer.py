import os
import sys
import argparse
from pathlib import Path
from typing import List

from openai import OpenAI
from pypdf import PdfReader


def ensure_api_key() -> None:
    """
    Ensure OPENAI_API_KEY is set in the environment.
    Exit with a clear message if it is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n\n"
            "Export it before running, e.g.:\n"
            '  export OPENAI_API_KEY="sk-..."\n'
        )
        sys.exit(1)


def get_client() -> OpenAI:
    """
    Create and return an OpenAI client.
    Assumes OPENAI_API_KEY is already set.
    """
    return OpenAI()


def read_pdf_text(pdf_path: Path) -> str:
    """
    Read all text from a PDF file.
    Returns an empty string if something goes wrong.
    """
    try:
        reader = PdfReader(str(pdf_path))
        chunks: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"  [ERROR] Failed to read PDF {pdf_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
    """
    Split a long text into overlapping chunks so we can safely send them to the model.

    chunk_size: max characters per chunk
    overlap:    number of characters to overlap between chunks to preserve context
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    end = chunk_size

    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        # move forward but keep some overlap
        start = end - overlap
        end = start + chunk_size

    return chunks


def summarize_chunk(
    client: OpenAI,
    chunk: str,
    model: str,
    style: str,
) -> str:
    """
    Summarize a single chunk of text.
    We keep this short because it will be combined later.
    """
    style_instruction = {
        "default": "Use a clear, student-friendly tone.",
        "bullet": "Focus on bullet points only.",
        "narrative": "Write in a smooth narrative style.",
        "executive": "Write in an executive-style summary for busy leaders.",
    }.get(style, "Use a clear, student-friendly tone.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise assistant that summarizes educational documents. "
                    f"{style_instruction}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following part of a document in 5–8 bullet points. "
                    "Keep it compact but capture all key ideas.\n\n"
                    f"{chunk[:12000]}"
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def combine_chunk_summaries(
    client: OpenAI,
    chunk_summaries: List[str],
    model: str,
    style: str,
    title: str,
) -> str:
    """
    Combine individual chunk summaries into a single, structured document-level summary.
    """
    style_instruction = {
        "default": "Use a clear, student-friendly tone.",
        "bullet": "Focus heavily on structured bullet points.",
        "narrative": "Write in a smooth narrative style.",
        "executive": "Write an executive-style summary with key risks, insights, and actions.",
    }.get(style, "Use a clear, student-friendly tone.")

    joined_summaries = "\n\n---\n\n".join(chunk_summaries)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that turns multiple partial summaries into one clear, "
                    "structured summary a college student can use to study."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document title: {title}\n\n"
                    "You are given partial summaries of different parts of a longer document.\n\n"
                    "TASK:\n"
                    "1) Read all the partial summaries.\n"
                    "2) Produce a single, well-structured summary with this schema:\n"
                    "   - Title\n"
                    "   - 5–7 sentence overview\n"
                    "   - 5–10 bullet key ideas\n"
                    "   - Optional: 3–5 action items or next steps\n\n"
                    f"Write in this style: {style_instruction}\n\n"
                    "Here are the partial summaries:\n\n"
                    f"{joined_summaries[:24000]}"
                ),
            },
        ],
        temperature=0.25,
    )

    return response.choices[0].message.content.strip()


def summarize_document(
    client: OpenAI,
    pdf_path: Path,
    model: str,
    style: str,
    max_chars_per_chunk: int,
) -> str:
    """
    Full pipeline for a single PDF:
    - Read text
    - Chunk if needed
    - Summarize each chunk
    - Combine into final summary (Markdown)
    """
    text = read_pdf_text(pdf_path)
    if not text.strip():
        raise ValueError("No text extracted from PDF.")

    chunks = chunk_text(text, chunk_size=max_chars_per_chunk, overlap=500)

    if len(chunks) == 1:
        # simple case: one call
        single_summary = combine_chunk_summaries(
            client=client,
            chunk_summaries=[chunks[0]],
            model=model,
            style=style,
            title=pdf_path.stem,
        )
        return single_summary

    # multi-chunk: summarize each chunk then combine
    chunk_summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"    Summarizing chunk {idx}/{len(chunks)}...")
        summary = summarize_chunk(client=client, chunk=chunk, model=model, style=style)
        chunk_summaries.append(summary)

    final_summary = combine_chunk_summaries(
        client=client,
        chunk_summaries=chunk_summaries,
        model=model,
        style=style,
        title=pdf_path.stem,
    )

    return final_summary


def process_pdfs(
    input_folder: Path,
    output_folder: Path,
    model: str,
    style: str,
    max_chars_per_chunk: int,
    force: bool,
) -> None:
    """
    Main loop:
    - Find PDFs in input_folder
    - For each PDF, create a Markdown summary in output_folder
    - Skip already summarized files unless --force is used
    """
    input_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)

    pdf_files = sorted(input_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_folder.resolve()}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {input_folder.resolve()}")
    ensure_api_key()
    client = get_client()

    for idx, pdf_path in enumerate(pdf_files, start=1):
        output_path = output_folder / f"{pdf_path.stem}.summary.md"

        print(f"\n[{idx}/{len(pdf_files)}] Processing {pdf_path.name}...")

        if output_path.exists() and not force:
            print(f"  Skipping (summary already exists at {output_path.name}). Use --force to regenerate.")
            continue

        try:
            summary = summarize_document(
                client=client,
                pdf_path=pdf_path,
                model=model,
                style=style,
                max_chars_per_chunk=max_chars_per_chunk,
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Summary for {pdf_path.name}\n\n")
                f.write(summary)
            print(f"  ✅ Summary saved to {output_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to summarize {pdf_path.name}: {e}")
            # continue with next file


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Summarize all PDFs in a folder using the OpenAI API."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="docs",
        help="Folder containing PDF files (default: docs)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="summaries",
        help="Folder to write Markdown summaries to (default: summaries)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="default",
        choices=["default", "bullet", "narrative", "executive"],
        help="Summary style (default, bullet, narrative, executive)",
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=6000,
        help="Maximum characters per chunk when splitting large PDFs (default: 6000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate summaries even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    process_pdfs(
        input_folder=input_folder,
        output_folder=output_folder,
        model=args.model,
        style=args.style,
        max_chars_per_chunk=args.max_chars_per_chunk,
        force=args.force,
    )


if __name__ == "__main__":
    main()
