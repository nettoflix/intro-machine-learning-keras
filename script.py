#https://learn.microsoft.com/pt-br/training/modules/intro-machine-learning-keras/3-neural-network
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import gzip
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

labels_map = {
  0: 'T-Shirt',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle Boot',
}
print("OLA MARILENE")

#Now that you have the data, you can display a sampling of images and corresponding labels from the training data.
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

#Now that you have the data, you can display a sampling of images and corresponding labels from the training data.

figure = plt.figure(figsize=(6, 6))
cols = 4
rows = 4

for i in range(1, cols * rows + 1):
  sample_idx = random.randint(0, len(training_images))
  image = training_images[sample_idx]
  label = training_labels[sample_idx]
  figure.add_subplot(rows, cols, i)
  plt.title(labels_map[label])
  plt.axis('off')
  plt.imshow(image.squeeze(), cmap='gray')

""" For such a small dataset, you could just use the NumPy arrays given by Keras to train the neural network. 
However, if you had a large dataset, you would need to wrap it in a tf.data.Dataset instance, 
which handles large data better by making it easy to keep just a portion of it in memory. 
You can wrap your data in a Dataset in this sample, so you're prepared to work with large data in the future. """
train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))


""" You saw earlier that each pixel of the image is represented by an unsigned int. 
In machine learning, you generally want the pixel values of your training data 
to be floating-point numbers between 0 and 1
so you convert them in the following way: """
train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

batch_size = 64
train_dataset = train_dataset.batch(batch_size).shuffle(500)
test_dataset = test_dataset.batch(batch_size).shuffle(500)





model = NeuralNetwork()
learning_rate = 0.1


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate)
metrics = ['accuracy']
model.compile(optimizer, loss_fn, metrics)

epochs = 50
print('\nFitting:')
model.fit(train_dataset, epochs=epochs)

print('\nEvaluating:')
(test_loss, test_accuracy) = model.evaluate(test_dataset)
print(f'\nTest accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')
model.save('outputs/model')
plt.show()
