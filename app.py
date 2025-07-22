from flask import Flask, request, jsonify
from src.predict import TFImageClassifier, TorchImageClassifier
import os

app = Flask(__name__)

TFclassifier = TFImageClassifier('model/bestmodel.keras')
Torchclassifier = TorchImageClassifier('model/bestmodel.pth')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400


    image_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(image_path)

    try:
        tf_label, tf_conf = TFclassifier.predict(image_path)
        torch_label, torch_conf = Torchclassifier.predict(image_path)

        result = {
            'tensorflow': {
                'label': tf_label,
                'confidence': round(float(tf_conf), 4)
            },
            'pytorch': {
                'label': torch_label,
                'confidence': round(float(torch_conf), 4)
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        os.remove(image_path)

if __name__ == '__main__':
    app.run(debug=True)
