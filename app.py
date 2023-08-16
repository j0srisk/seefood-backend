import os
from flask import Flask, request, jsonify
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")


@app.route("/predict", methods=["POST"])
def predict():
    provided_api_key = request.headers.get("X-API-Key")
    if provided_api_key == API_KEY:
        data = request.json
        url = data["url"]
        question = data["question"]

        image = Image.open(requests.get(url, stream=True).raw)

        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )

        encoding = processor(image, question, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "Unauthorized"}), 401
