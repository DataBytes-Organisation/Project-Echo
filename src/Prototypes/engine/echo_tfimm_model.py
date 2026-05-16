import torch
import timm
import math
import tfimm
import tensorflow as tf 

########################################################################################
# MODEL PARAMETERS
########################################################################################
MODEL_INPUT_IMAGE_WIDTH = 224
MODEL_INPUT_IMAGE_HEIGHT = 224
MODEL_INPUT_IMAGE_CHANNELS = 1
MODEL_OUTPUT_CLASSES = 15


########################################################################################
# CLASSIFIER LAYER
########################################################################################
class EchoClassifierLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EchoClassifierLayer, self).__init__()
        
        dropout=0.5
        
        self.fc1 = tf.keras.layers.Dense(128, 
                                         kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                         activation=tf.keras.activations.relu)

        self.fc2 = tf.keras.layers.Dense(128, 
                                         kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                         activation=tf.keras.activations.relu)
        
        self.do2 = tf.keras.layers.Dropout(dropout)        
        
        self.out = tf.keras.layers.Dense(MODEL_OUTPUT_CLASSES, 
                                         activation=tf.keras.activations.linear)

    def call(self, inputs):
        x = self.fc1(inputs)               
        x = self.fc2(x)               
        x = self.do2(x)           
        x = self.out(x)
        return x


########################################################################################
# CLASSIFIER MODEL - leveraging EfficientNetV2
########################################################################################
class EchoTfimmModel(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):  
        super(EchoTfimmModel, self).__init__(*args, **kwargs)
        
        self.fm = tfimm.create_model("efficientnet_v2_s_in21k", pretrained=True, in_channels=MODEL_INPUT_IMAGE_CHANNELS)
        self.flat = tf.keras.layers.Flatten() 
        self.classifier = EchoClassifierLayer()

    def call(self, inputs, training=False):  
        x = self.fm.forward_features(inputs) 
        x = self.flat(x)
        x = self.classifier(x)               
        return x