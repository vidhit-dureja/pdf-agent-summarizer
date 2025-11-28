import streamlit as st
from summarizer import summarize_pdf_bytes

st.title("PDF Agent Summarizer (MVP)")

st.write("Upload one or more PDFs and get structured Markdown summaries using OpenAI.")

uploaded_files = st.file_uploader(
    "Choose PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

style = st.selectbox(
    "Summary style",
    ["default", "bullet", "narrative", "executive"],
    index=0,
)

model = st.text_input("OpenAI model", value="gpt-4.1-mini")

max_chars = st.number_input(
    "Max characters per chunk",
    min_value=1000,
    max_value=12000,
    value=6000,
    step=500,
)

if st.button("Summarize") and uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Summarizing {uploaded_file.name}..."):
            pdf_bytes = uploaded_file.read()
            try:
                summary = summarize_pdf_bytes(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                    model=model,
                    style=style,
                    max_chars_per_chunk=max_chars,
                )
                st.success(f"Done: {uploaded_file.name}")
                st.markdown(f"### Summary: {uploaded_file.name}")
                st.markdown(summary)

                st.download_button(
                    label=f"Download {uploaded_file.name}.summary.md",
                    data=summary,
                    file_name=f"{uploaded_file.name}.summary.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Error with {uploaded_file.name}: {e}")
