import tensorflow as tf
from tensorflow.keras import layers

class BinaryClassifier(tf.Module):
    def __init__(self, name):
        super().__init__(name)
        self.dense = layers.Dense(1, activation='sigmoid')

    @tf.function
    def __call__(self, x):
        return self.dense(x)