from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.parse
import re
import os

app = Flask(__name__)
CORS(app)

class PhishingDetector:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def preprocess_url(self, url: str) -> str:
        try:
            decoded_url = urllib.parse.unquote(url)
        except:
            decoded_url = url

        parsed_url = urllib.parse.urlparse(decoded_url)

        features = [
            parsed_url.scheme,                 # Protocol
            parsed_url.netloc,                 # Domain
            parsed_url.path,                   # Path
            str(len(parsed_url.netloc)),       # Domain length
            str(len(parsed_url.path)),         # Path length
            str(url.count('/')),               # Number of slashes
            str(len(url)),                     # Total URL length
            str(url.count('.')),               # Number of dots
            str('@' in url),                   # Presence of @ symbol
            str('https' in url.lower()),       # Use of HTTPS
            str(bool(re.search(r'\d', url))),  # Presence of digits
            str(bool(re.search(r'[/@?=&#]', url))),  # Special characters
            str(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url))),  # IP address
            str(bool(re.search(r'[^a-zA-Z0-9.-]', url))),  # Non-alphanumeric characters
            str(url.count('-')),               # Number of hyphens
        ]

        return ' '.join(features)

    def predict(self, url: str) -> dict:
        processed_url = self.preprocess_url(url)
        sequence = self.tokenizer.texts_to_sequences([processed_url])
        padded_sequence = pad_sequences(sequence, maxlen=150, padding='post', truncating='post')
        prediction = self.model.predict(padded_sequence)[0][0]
        return {
            "probability": float(prediction),
            "is_phishing": prediction > 0.5,
            "label": "Phishing" if prediction > 0.5 else "Legitimate"
        }

# Initialize the Phishing Detector
detector = PhishingDetector(
    model_path='/enhanced_phishing_model.h5',
    tokenizer_path='/enhanced_url_tokenizer.pickle'
)

@app.route('/predict', methods=['GET'])
def predict():
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        prediction = detector.predict(url)
        response = {
            "url": url,
            "prediction": prediction["label"],
            "probability": prediction["probability"]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
