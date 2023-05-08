from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_kehumasan = joblib.load('./models/kehumasan/model_kehumasan.pkl')
vectorizer_kehumasan = joblib.load('./models/kehumasan/vectorizer_kehumasan.pkl')

model_pelayanan = joblib.load('./models/pelayanan/model_pelayanan.pkl')
vectorizer_pelayanan = joblib.load('./models/pelayanan/vectorizer_pelayanan.pkl')

model_penyuluhan = joblib.load('./models/penyuluhan/model_penyuluhan.pkl')
vectorizer_penyuluhan = joblib.load('./models/penyuluhan/vectorizer_penyuluhan.pkl')

model_umum = joblib.load('./models/umum/model_umum.pkl')
vectorizer_umum = joblib.load('./models/umum/vectorizer_umum.pkl')

@app.route('/api/public-relations', methods=['POST'])
def predict_public_relations():
    try:
        text = request.json['text']
        text_counts = vectorizer_kehumasan.transform([text])
        predicted_label = model_kehumasan.predict(text_counts)[0]
        predicted_label = int(predicted_label)
        return jsonify({'sentiment': predicted_label, 'text': text})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/service', methods=['POST'])
def predict_service():
    try:
        text = request.json['text']
        text_counts = vectorizer_pelayanan.transform([text])
        predicted_label = model_pelayanan.predict(text_counts)[0]
        predicted_label = int(predicted_label)
        return jsonify({'sentiment': predicted_label, 'text': text})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/api/counseling', methods=['POST'])
def predict_counseling():
    try:
        text = request.json['text']
        text_counts = vectorizer_penyuluhan.transform([text])
        predicted_label = model_penyuluhan.predict(text_counts)[0]
        predicted_label = int(predicted_label)
        return jsonify({'sentiment': predicted_label, 'text': text})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/api/general', methods=['POST'])
def predict_general():
    try:
        text = request.json['text']
        text_counts = vectorizer_umum.transform([text])
        predicted_label = model_umum.predict(text_counts)[0]
        predicted_label = int(predicted_label)
        return jsonify({'sentiment': predicted_label, 'text': text})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/')
def index():
    return 'Hello from Flask!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
