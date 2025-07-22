import requests
import sys
import matplotlib.pyplot as plt
from PIL import Image

url = 'http://localhost:5000/predict'
image_path = sys.argv[1]
if not image_path:
    print("Usage: python main.py path_to_image")
    sys.exit(1)

image = Image.open(image_path).convert('RGB')
plt.imshow(image)
plt.title("Image")
plt.axis('off')
plt.show()
with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print(response.json())