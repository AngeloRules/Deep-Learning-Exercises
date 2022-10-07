import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class LeNetLayer(layers.Layer):

    def __init__(self,channels,kernel_size,input_shape="",activation='relu'):
        super().__init__()
        self.conv1 = layers.Conv2D(channels,kernel_size,activation=activation,input_shape=input_shape)
        self.pool1 = layers.AveragePooling2D(pool_size=(2,2),strides=2)

    def call(self,input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool1(x)
        x = tf.nn.relu(x)
        return x 

class LeNet5(keras.Model):
    def __init__(self,classes=10):
        super().__init__()
        self.flatten = layers.Flatten()
        self.layer1 = LeNetLayer(channels=6,kernel_size=(5,5),input_shape=(32,32,1))
        self.layer2 = LeNetLayer(channels=16,kernel_size=(5,5))
        self.fully_connected1 = layers.Dense(120,activation='relu')
        self.fully_connected2 = layers.Dense(84,activation='relu')
        self.classifier =  layers.Dense(classes,activation='softmax')

    def call(self,input_tensor):
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        return self.classifier(x)

