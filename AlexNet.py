import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AlexNetLayer(layers.Layer):

    def __init__(self,channels,kernel_size,activation='relu',input_shape="",strides=1,padding="same"):
        super().__init__()
        self.conv1 = layers.Conv2D(channels,kernel_size,activation=activation,input_shape=input_shape,strides=strides,padding=padding)
        self.pool = layers.MaxPooling2D(pool_size=(3,3),strides=2)

    def call(self,input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool(x)
        x = tf.nn.relu(x)
        return x 
    
class AlexNet(keras.Model):
    
    def __init__(self,classes=10):
        super().__init__()

        self.flatten = layers.Flatten()
        self.layer1 = AlexNetLayer(channels=96,kernel_size=(11,11),input_shape=(224,224,1),strides=4)
        self.layer2 = AlexNetLayer(channels=256,kernel_size=(5,5),padding='same')
        self.layer3 = AlexNetLayer(channels=384,kernel_size=(3,3),padding='same')
        self.layer4 = layers.Conv2D(filters=384,kernel_size=(3,3),padding='same',activation='relu')
        self.layer5 = layers.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu')
        self.fully_connected1 = layers.Dense(9216,activation='relu')
        self.fully_connected2 = layers.Dense(4096,activation='relu')
        self.classifier =  layers.Dense(classes,activation='softmax')
    
    def call(self,input_tensor):
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        x = self.fully_connected2(x)
        return self.classifier(x)


