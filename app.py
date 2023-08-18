import os
import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

API_KEY = os.environ.get("API_KEY")

app = Flask(__name__)
CORS(app)

if torch.cuda.is_available():
    print("GPU is available and being used.")
else:
    print("No GPU available, using CPU.")


@app.route("/predict", methods=["POST"])
def predict():
    provided_api_key = request.headers.get("X-API-Key")
    if provided_api_key == API_KEY:
        start_time = time.time()

        question = request.form["question"]
        image_file = request.files["file"]

        image = Image.open(image_file)

        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )

        encoding = processor(image, question, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        end_time = time.time()

        processing_time = int((end_time - start_time) * 1000)

        return jsonify({"answer": answer, "processing_time_ms": processing_time})
    else:
        return jsonify({"error": "Unauthorized"}), 401
