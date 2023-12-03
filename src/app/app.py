from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

model_path = 'ruta/donde/guardar'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Preprocesar el texto
    inputs = tokenizer(text, return_tensors='pt')
    
    # Hacer la predicción
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener la predicción
    predicted_class = torch.argmax(outputs.logits).item()
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
