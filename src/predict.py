import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class TFImageClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)

        preds = self.model.predict(img_arr)
        class_index = np.argmax(preds[0])
        confidence = preds[0][class_index]

        return self.class_labels[class_index], confidence

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Dropout(0.3),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

class TorchImageClassifier:
    def __init__(self, model_path):
        self.class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])

        self.model = CNNModel(num_classes=len(self.class_labels))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.model.eval()

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            preds = F.softmax(outputs, dim=1)

        class_index = torch.argmax(preds[0]).item()
        confidence = preds[0][class_index].item()
        return self.class_labels[class_index], confidence
