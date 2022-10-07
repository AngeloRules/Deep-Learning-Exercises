import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from LeNet5 import LeNet5

(x_train,y_train),(x_test,y_test) =  mnist.load_data()

#x_train = x_train/255.0
#x_test = x_test/255.0

x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0

model = LeNet5(classes=10)

model.compile(
    optimizer='adam',
    metrics = ['accuracy'],
    loss = keras.losses.SparseCategoricalCrossentropy()
)

model.fit(x_train,y_train,epochs=2,batch_size=64)
print(model.summary())
