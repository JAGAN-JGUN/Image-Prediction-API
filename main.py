import sys
from src.predict import TFImageClassifier, TorchImageClassifier

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py path_to_image")
        sys.exit(1)

    TFclassifier = TFImageClassifier('model/bestmodel.keras')
    TFlabel, TFconfidence = TFclassifier.predict(sys.argv[1])

    print(f"Tensorflow model prediction: {TFlabel} (confidence: {TFconfidence:.2f})")

    Torchclassifier = TorchImageClassifier('model/bestmodel.pth')
    Torchlabel, Torchconfidence = Torchclassifier.predict(sys.argv[1])

    print(f"PyTorch model prediction: {Torchlabel} (confidence: {Torchconfidence:.2f})")

