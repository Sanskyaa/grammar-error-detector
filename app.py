
import streamlit as st
from pipeline import grammar_error_detector

st.set_page_config(page_title="Grammar Error Detector", page_icon="📝", layout="centered")

st.title("📝 Grammar Error Detector")
st.write("An NLP-based system that detects grammatical errors, classifies the error type, and suggests corrections.")

sentence = st.text_area("Enter a sentence:", height=100, placeholder="e.g. He go to school everyday.")

if st.button("Check Grammar"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        with st.spinner("Analyzing..."):
            result = grammar_error_detector(sentence)

        st.subheader("Result")

        if result["has_error"]:
            st.error("❌ Grammatical error detected")
            st.markdown(f"**Error Type(s):** {', '.join(result['error_types'])}")
            st.markdown(f"**Corrected Sentence:** {result['corrected_sentence']}")
        else:
            st.success("✅ No grammatical error detected")

        with st.expander("See raw output"):
            st.json(result)

st.markdown("---")
st.caption("NLP Mini Project — Grammar Error Detector | Logistic Regression + T5 + spaCy POS tagging")
