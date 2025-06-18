import os
import openai
import pandas as pd
import streamlit as st

# Load OpenAI key
openai.api_key = st.secrets["openai"]["api_key"]

# File uploader
st.title("🏆 Reward Dashboard with LLM Confidence")
uploaded_file = st.file_uploader("Upload a CSV file with a 'reward_reason' column")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "reward_reason" not in df.columns:
        st.error("CSV must contain a 'reward_reason' column.")
    else:
        # LLM classification function
        def classify_with_llm(text):
            prompt = (
                f"Classify the following reward reason into a category (like Teamwork, Leadership, Reliability) "
                f"and estimate your confidence (High, Medium, Low):\n\n"
                f"Reason: '{text}'\n\n"
                f"Respond in this format:\nCategory: <Category>\nConfidence: <High|Medium|Low>"
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                output = response["choices"][0]["message"]["content"]
                lines = output.strip().split("\n")
                category = lines[0].split(": ")[1]
                confidence = lines[1].split(": ")[1]
                return category, confidence
            except Exception as e:
                return "Unknown", "Low"

        # Classify using LLM
        with st.spinner("Classifying rewards with LLM..."):
            df[["category", "confidence_category"]] = df["reward_reason"].apply(
                lambda x: pd.Series(classify_with_llm(x))
            )

        st.success("Classification complete!")
        st.dataframe(df)

        # Optional: Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV with Categories", csv, "classified_rewards.csv", "text/csv")
