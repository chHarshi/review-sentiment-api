"""
Small Gradio UI that calls the already-deployed sentiment API.
Does NOT load the model itself — keeps this app lightweight.
"""

import os
import requests
import gradio as gr

API_URL = "https://review-sentiment-api-f8dq.onrender.com/predict"


def classify_review(text):
    if not text.strip():
        return "Please enter a review first."

    try:
        response = requests.post(API_URL, json={"text": text}, timeout=60)
        response.raise_for_status()
        result = response.json()

        label = result["label"]
        confidence = result["confidence"]

        emoji = "😊" if label == "positive" else "😞"
        return f"{emoji} **{label.upper()}** (confidence: {confidence*100:.1f}%)"

    except requests.exceptions.Timeout:
        return "The API is waking up from sleep (free-tier cold start) — please try again in 30-60 seconds."
    except Exception as e:
        return f"Something went wrong: {e}"


demo = gr.Interface(
    fn=classify_review,
    inputs=gr.Textbox(
        label="Product Review",
        placeholder="e.g. This yogurt was creamy and delicious, will buy again!",
        lines=4,
    ),
    outputs=gr.Markdown(label="Sentiment"),
    title="FMCG Review Sentiment Classifier",
    description=(
        "A fine-tuned DistilBERT model classifies product reviews as positive or negative. "
        "Note: binary classification only — mixed/neutral reviews will be forced into one category. "
        "First request may take 30-60s if the backend API is waking up from sleep."
    ),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)