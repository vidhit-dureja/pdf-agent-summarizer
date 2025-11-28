import streamlit as st
from summarizer import summarize_pdf_bytes

st.title("PDF Agent Summarizer (MVP)")

st.write("Upload a PDF and get a structured Markdown summary using OpenAI.")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

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

if st.button("Summarize") and uploaded_file is not None:
    with st.spinner("Summarizing..."):
        pdf_bytes = uploaded_file.read()
        try:
            summary = summarize_pdf_bytes(
                pdf_bytes=pdf_bytes,
                filename=uploaded_file.name,
                model=model,
                style=style,
                max_chars_per_chunk=max_chars,
            )
            st.success("Done!")
            st.markdown("### Summary")
            st.markdown(summary)

            st.download_button(
                label="Download summary as .md",
                data=summary,
                file_name=f"{uploaded_file.name}.summary.md",
                mime="text/markdown",
            )
        except Exception as e:
            st.error(f"Error: {e}")
