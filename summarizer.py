from pathlib import Path
from typing import List
from openai import OpenAI
from pypdf import PdfReader
from io import BytesIO

def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        end = start + chunk_size
    return chunks

def summarize_chunk(
    client: OpenAI,
    chunk: str,
    model: str,
    style: str,
) -> str:
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

def summarize_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    model: str = "gpt-4.1-mini",
    style: str = "default",
    max_chars_per_chunk: int = 6000,
) -> str:
    client = OpenAI()
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
            title=Path(filename).stem,
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
        title=Path(filename).stem,
    )
    return final_summary
