from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle
import os
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

app = Flask(__name__)


MODEL_TYPE = "BiLSTM"  # choose best model

if MODEL_TYPE == "BiLSTM":
    model = tf.keras.models.load_model("best_bilstm_model.h5")
    with open("bilstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    MAX_LEN = 250

    def predict_sentiment(text):
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        pred = model.predict(padded)
        label = 1 if pred > 0.5 else 0
        return label_encoder.inverse_transform([label])[0]

else:  # DistilBERT
    model_path = "best_distilbert_model"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    def predict_sentiment(text):
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        outputs = model(inputs)
        logits = outputs.logits
        pred = np.argmax(tf.nn.softmax(logits, axis=-1), axis=1)
        return label_encoder.inverse_transform(pred.numpy())[0]


# Flask Routes

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['review_text']
    prediction = predict_sentiment(text)
    return render_template('index.html', review_text=text, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
