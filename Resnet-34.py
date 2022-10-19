import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Conv2D(layers.Layer):

    def __init__(self,kernel_size,padding,strides,filters,activation):
        super().__init__()
        self.conv = layers.Conv2D(
            kernel_size=kernel_size,padding=padding,
            strides=strides,filters=filters,activation=activation
        )
        self.batch_norm = layers.BatchNormalization()

    def call(self,input_tensor,training=True):
        x = self.conv(input_tensor)
        x = self.batch_norm(x,training)
        return x 


class ResNetBlock(layers.Layer):

    def __init__(self,channels,stride=1):
        super().__init__()

        self.skip_dotted = (stride != 1)
        self.conv1 = Conv2D((3,3),'same',stride,channels,'relu')
        self.conv2 = Conv2D(kernel_size=(3,3),padding='same',filters=channels,activation='relu',strides=1)
        if self.skip_dotted:
            self.conv3 = Conv2D(filters=channels,kernel_size=(1,1),strides=stride,padding='valid',activation='relu')

    
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if self.skip_dotted:
            input_tensor = self.conv3(input_tensor)
        x = layers.add([input_tensor, x])
        x = tf.nn.relu(x)
        return x 


class ResNet34(keras.Model):

    def __init__(self,num_classes=5):
        super().__init__()
        self.conv1  = Conv2D(filters=64,kernel_size=(7,7),strides=2,padding='same',activation='relu')
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(pool_size=(3,3),strides=2)

        self.block1_1 = ResNetBlock(64)
        self.block1_2 = ResNetBlock(64)
        self.block1_3 = ResNetBlock(64)

        self.block2_1 = ResNetBlock(128,2)
        self.block2_2 = ResNetBlock(128)
        self.block2_3 = ResNetBlock(128)
        self.block2_4 = ResNetBlock(128)
    
        self.block3_1 = ResNetBlock(256,2)
        self.block3_2 = ResNetBlock(256)
        self.block3_3 = ResNetBlock(256)
        self.block3_4 = ResNetBlock(256)
        self.block3_5 = ResNetBlock(256)
        self.block3_6 = ResNetBlock(256)

        self.block4_1 = ResNetBlock(512,2)
        self.block4_2 = ResNetBlock(512)
        self.block4_3 = ResNetBlock(512)

        self.pool2 = layers.GlobalAveragePooling2D()
        self.fully_connected1 = layers.Dense(512,activation='relu')
        self.fully_connected2 = layers.Dense(512,activation='relu')
        self.drop_out1 = layers.Dropout(0.5)
        self.drop_out2 = layers.Dropout(0.5)
        self.classifier = layers.Dense(units=num_classes,activation='softmax') 


    def call(self,input_tensor,training=True):
        x = self.conv1(input_tensor)
        x = self.batch_norm(x,training)
        x = self.relu(x)
        x = self.pool1(x)
        #64 filter section
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        #128 filter section
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)

        #256 filter section
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        x = self.block3_6(x)

        #512 filter section
        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.pool2(x)
        
        #fully connected layer
        x = self.fully_connected1(x)
        x = self.drop_out1(x)
        x = self.fully_connected2(x)
        x = self.drop_out2(x)
        x = self.classifier(x)
        return x









