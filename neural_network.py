import tensorflow as tf
"""Because each image has 28 × 28 = 784 pixels, we need 784 nodes in the input layer (one for each pixel value). 
Add one hidden layer with 20 nodes and a ReLU (rectified linear unit) activation function. 
You want the output of your network to be a vector of size 10, therefore your output layer 
needs to have 10 nodes. Here's the Keras code that defines this neural network:"""
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

class NeuralNetwork(tf.keras.Model):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.sequence = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(20, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime
  
def get_data():
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
    test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

    batch_size = 64
    train_dataset = train_dataset.batch(batch_size).shuffle(500)
    test_dataset = test_dataset.batch(batch_size).shuffle(500)
    return train_dataset, test_dataset