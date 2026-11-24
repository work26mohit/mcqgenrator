import os
import json
import traceback
import pandas as pd
import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

# 1. Load the Response.json template from your project directory
with open("Response.json", "r") as file:
    RESPONSE_JSON = json.load(file)

# 2. App Title
st.title("MCQs Creator Application with LangChain & Gemini")

# 3. Create Form with Inputs and Submit Button
with st.form("user_inputs"):
    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    # Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50, value=5)
    subject = st.text_input("Insert Subject", max_chars=50, placeholder="e.g., Biology")
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")

    # Submit Button
    button = st.form_submit_button("Create MCQs")

# 4. Handle Form Submission
if button:
    if uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Generating MCQs..."):
            try:
                # Extract text from uploaded PDF/TXT using utils.py
                text = read_file(uploaded_file)

                # Run the Gemini LCEL chain and track token usage
                with get_usage_metadata_callback() as cb:
                    response = generate_evaluate_chain.invoke(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )

                # Print token usage in terminal/logs
                total_tokens = sum(m.get("total_tokens", 0) for m in cb.usage_metadata.values())
                print(f"Total Tokens Used: {total_tokens}")

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error(f"Error generating MCQs: {e}")

            else:
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.subheader("Generated MCQs")
                            st.table(df)

                            # Display the expert review in a text area
                            st.text_area(label="Review", value=response.get("review", ""), height=150)

                            # Optional: Download button for the generated CSV
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download MCQs as CSV",
                                data=csv,
                                file_name=f"{subject.lower()}_mcqs.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("Error parsing the generated quiz table data.")
                else:
                    st.write(response)
    else:
        st.warning("Please upload a file and fill in all the fields before submitting.")