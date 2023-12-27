import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import gzip
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, labels_map
from PIL import Image
import requests

model = tf.keras.models.load_model('outputs/model')
url = 'https://raw.githubusercontent.com/MicrosoftDocs/tensorflow-learning-path/main/intro-keras/predict-image.png'

with Image.open(requests.get(url, stream=True).raw) as image:
  xtest = np.asarray(image, dtype=np.float32)
  X = np.asarray(image, dtype=np.float32).reshape((-1, 28, 28)) / 255.0


predicted_vector = model.predict(X)
predicted_index = np.argmax(predicted_vector)
predicted_name = labels_map[predicted_index]

print(f'Predicted class: {predicted_name}')

probs = tf.nn.softmax(predicted_vector.reshape((-1,)))
for i,p in enumerate(probs):
    print(f'{labels_map[i]} -> {p:.3f}')




plt.figure()
plt.axis('off')
plt.imshow(X.squeeze(), cmap='gray')
plt.show()