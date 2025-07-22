# Image Prediction API

This repository provides a Flask-based API to perform image classification using a pre-trained model.

The model was trained using code from a separate repository:  
🔗 [Image-Classification](https://github.com/JAGAN-JGUN/Image-Classification)

---

## Model Training Overview

The models served by this API were trained using a deep learning pipeline for **Intel Image Classification**, developed in both **TensorFlow** and **PyTorch**.

### Intel Image Classification Dataset

This dataset used to train the models is provided by **Intel** and is designed for natural scene classification. It contains RGB images across 6 natural categories:

* **Buildings**

* **Forest**

* **Glacier**

* **Mountain**

* **Sea**

* **Street**

### Training Highlights

- **TensorFlow Model**
  - Data Augmentation: RandomFlip, RandomZoom, RandomRotation
  - CNN with BatchNorm, Dropout, and GlobalAveragePooling2D
  - Regularization via L1L2, EarlyStopping, ModelCheckpoint
- **PyTorch Model**
  - CNN with BatchNorm, Dropout, AdaptiveAvgPool2d
  - Best model checkpointed with torch.save

### Evaluation Results

| Framework   | Train Acc | Val Acc | Train Loss | Val Loss |
|-------------|-----------|---------|------------|----------|
| TensorFlow  | 84.25%    | 83.0%   | 0.5474     | 0.5870   |
| PyTorch     | 87.01%    | 87.1%   | 0.3631     | 0.3473   |

Evaluation included:
- Classification Reports
- Confusion Matrices using Seaborn + Matplotlib

---

## Project Structure

```
.
├── app.py                  # Flask app for serving image prediction API
├── main.py                 # Main script (could be CLI or init script)
├── request.py              # Internal/external request handler
├── model/
│   ├── bestmodel.keras     # TensorFlow model
│   └── bestmodel.pth       # PyTorch model
├── src/
│   ├── predict.py          # Prediction logic
│   └── __init__.py         # Python module initializer
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/JAGAN-JGUN/image-prediction-api.git
cd image-prediction-api
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Requirements

Install dependencies manually (if `requirements.txt` is missing):

```bash
pip install flask torch tensorflow pillow
```

### 4. Run the Flask App

```bash
python app.py
```

Visit: `http://localhost:5000`

---

## Model Information

Both TensorFlow and PyTorch models are available:
- `model/bestmodel.keras`
- `model/bestmodel.pth`

Prediction logic is managed in `src/predict.py`. Switch between models based on your use case.

---

## Example Prediction via curl

```bash
curl -X POST -F image=@sample.jpg http://localhost:5000/predict
```

---

## Example prediction via CLI Script (request.py)

You can also make predictions directly from the command line using the **request.py** script. This is useful for testing locally without using the web UI.

### Usage

```bash
python request.py path/to/image.jpg #Replace path/to/image.jpg with your image path.
```

This sends a POST request to the Flask API running at **http://localhost:5000/predict**.

It displays the selected image and prints the prediction result as **JSON**.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Training code adapted from: [Image-Classification](https://github.com/JAGAN-JGUN/Image-Classification)
- Built with Flask, PyTorch, and TensorFlow.
