import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VGGNetLayer1(layers.Layer):

    def __init__(self,channels,kernel_size,padding,pool_size,pool_stride,activation='relu',input_shape=''):
        super().__init__()
        self.conv1 = layers.Conv2D(filters=channels,kernel_size=kernel_size,padding=padding,activation=activation,input_shape=input_shape)
        self.conv2 = layers.Conv2D(filters=channels,kernel_size=kernel_size,padding=padding,activation=activation)
        self.pool1 = layers.MaxPooling2D(pool_size=pool_size,strides=pool_stride)

    def call(self,input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.pool1(x)
        return tf.nn.relu(x)

class VGGNetLayer2(layers.Layer):

    def __init__(self,channels,kernel_size,padding,pool_size,pool_stride,activation='relu',input_shape=''):
        super().__init__()
        self.block1 = layers.Conv2D(filters=channels,kernel_size=kernel_size,padding=padding,activation=activation,input_shape=input_shape)
        self.block2 = layers.Conv2D(filters=channels,kernel_size=kernel_size,padding=padding,activation=activation)
        self.block3 = layers.Conv2D(filters=channels,kernel_size=kernel_size,padding=padding,activation=activation)
        self.pool2 = layers.MaxPooling2D(pool_size=pool_size,strides=pool_stride)
    
    def call(self,input_tensor):
        x = self.block1(input_tensor)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool2(x)
        return tf.nn.relu(x)


class VGGNet_16(keras.Model):

    def __init__(self,num_classes=1000):
        super().__init__()
        self.layer1 = VGGNetLayer1(input_shape=(224,224,3),channels=64,kernel_size=(3,3),padding='same',activation='relu',pool_size=(2,2),pool_stride=(2,2))
        self.layer2 = VGGNetLayer1(input_shape=(),channels=128,kernel_size=(3,3),padding='same',activation='relu',pool_size=(2,2),pool_stride=(2,2))
        self.layer3 = VGGNetLayer2(input_shape=(),channels=256,kernel_size=(3,3),padding='same',activation='relu',pool_size=(2,2),pool_stride=(2,2))
        self.layer4 = VGGNetLayer2(input_shape=(),channels=512,kernel_size=(3,3),padding='same',activation='relu',pool_size=(2,2),pool_stride=(2,2))
        self.layer5 = VGGNetLayer2(input_shape=(),channels=512,kernel_size=(3,3),padding='same',activation='relu',pool_size=(2,2),pool_stride=(2,2))
        self.flatten = layers.Flatten()
        self.fully_connected1 = layers.Dense(4096,activation='relu')
        self.fully_connected2 = layers.Dense(4096,activation='relu')
        self.classfier = layers.Dense(num_classes,activation='softmax')

    def call(self,input_tensor):
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        return self.classfier(x)