# summarizer.py
from pathlib import Path
from typing import List
from openai import OpenAI
from pypdf import PdfReader

# paste your ensure_api_key, get_client, read_pdf_text,
# chunk_text, summarize_chunk, combine_chunk_summaries here

def summarize_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    model: str = "gpt-4.1-mini",
    style: str = "default",
    max_chars_per_chunk: int = 6000,
) -> str:
    client = OpenAI()
    # save to a temp path in memory-like style
    temp_path = Path(filename)
    # pypdf can read from a BytesIO, but to keep it simple we reuse read_pdf_text:
    from io import BytesIO
    reader = PdfReader(BytesIO(pdf_bytes))
    text_chunks: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            text_chunks.append(text)
    text = "\n\n".join(text_chunks)

    if not text.strip():
        raise ValueError("No text extracted from PDF.")

    chunks = chunk_text(text, chunk_size=max_chars_per_chunk, overlap=500)

    if len(chunks) == 1:
        return combine_chunk_summaries(
            client=client,
            chunk_summaries=[chunks[0]],
            model=model,
            style=style,
            title=temp_path.stem,
        )

    chunk_summaries: List[str] = []
    for chunk in chunks:
        summary = summarize_chunk(client=client, chunk=chunk, model=model, style=style)
        chunk_summaries.append(summary)

    final_summary = combine_chunk_summaries(
        client=client,
        chunk_summaries=chunk_summaries,
        model=model,
        style=style,
        title=temp_path.stem,
    )
    return final_summary
