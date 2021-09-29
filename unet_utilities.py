import os
import numpy as np
from math import inf
import cv2
import tifffile as tiff
import h5py
from scipy.spatial import distance
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from data_generators import *

"""
Deep learning/TF-related operations.
"""

def safe_log(tensor):
    """
    Prevents log(0)

    Arguments:
    * tensor - tensor
    """
    return tf.log(tf.clip_by_value(tensor,1e-32,tf.reduce_max(tensor)))

class UNetConvLayer(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetConvLayer,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer = keras.Sequential()
        if self.factorization == False:
            self.layer.add(keras.layers.Conv2D(
                self.depth,self.conv_size,strides=self.stride,
                padding=self.padding,
                kernel_regularizer=keras.regularizers.l2(self.beta)))
        
        if self.factorization == True:
            self.layer.add(keras.layers.Conv2D(
                self.depth,[1,self.conv_size],
                strides=self.stride,padding=self.padding,
                kernel_regularizer=keras.regularizers.l2(self.beta)))
            self.layer.add(keras.layers.Conv2D(
                self.depth,[self.conv_size,1],
                strides=self.stride,padding=self.padding,
                kernel_regularizer=keras.regularizers.l2(self.beta)))
        self.layer.add(keras.layers.LeakyReLU())
        self.layer.add(keras.layers.Dropout(self.dropout_rate))
    
    def call(self,x):
        return self.layer(x)

class UNetConvBlock(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride=1,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetConvBlock,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer = keras.Sequential()
        self.layer.add(UNetConvLayer(
            self.depth,self.conv_size,self.stride,self.factorization,
            self.padding,self.dropout_rate,self.beta))
        self.layer.add(UNetConvLayer(
            self.depth,self.conv_size,1,self.factorization,
            self.padding,self.dropout_rate,self.beta))
    
    def call(self,x):
        return self.layer(x)

class UNetReductionBlock(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride=1,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetReductionBlock,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer_pre = UNetConvBlock(
            self.depth,self.conv_size,self.stride,self.factorization,
            self.padding,self.dropout_rate)
        self.layer_down = keras.layers.MaxPool2D([2,2],2)
    
    def call(self,x):
        pre = self.layer_pre(x)
        down = self.layer_down(pre)
        return down,pre

class UNetReconstructionBlock(keras.layers.Layer):
    def __init__(self,
                 depth,
                 conv_size,
                 stride=1,
                 factorization=False,
                 padding='VALID',
                 dropout_rate=0.2,
                 beta=0.005):
        
        super(UNetReconstructionBlock,self).__init__()
        self.depth = depth
        self.conv_size = conv_size
        self.stride = stride
        self.factorization = factorization
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.setup_layers()

    def setup_layers(self):
        self.layer_pre = keras.Sequential()
        self.layer_pre.add(keras.layers.Conv2DTranspose(
            self.depth,3,strides=2,padding='same',
            kernel_regularizer=keras.regularizers.l2(self.beta)))
        self.concat_op = keras.layers.Concatenate(axis=-1)
        self.layer_post = UNetConvBlock(
            self.depth,self.conv_size,self.stride,self.factorization,
            self.padding,self.dropout_rate,self.beta)
    
    def crop_x_to_y(self,x,y):
        fraction = x.shape[2] / y.shape[2]
        if fraction == 1.:
            return x
        else:
            return tf.image.central_crop(x,fraction)

    def call(self,x,y=None):
        x = self.layer_pre(x)
        if y is not None:
            y = self.crop_x_to_y(y,x)
        if y is not None:
            x = self.concat_op([x,y])
        return self.layer_post(x)

class ChannelSqueezeAndExcite(keras.layers.Layer):
    def __init__(self,n_channels,beta):
        self.n_channels = n_channels
        self.setup_layers()
        self.beta = beta

    def setup_layers(self):
        self.layer = keras.Sequential([
            keras.layers.Dense(self.n_channels),
            keras.layers.Activation('relu'),
            keras.layers.Dense(self.n_channels),
            keras.layers.Activation('sigmoid')])
    
    def call(self,x):
        squeezed_input = tf.math.reduce_mean(x,[1,2])
        excited_input = self.layer(squeezed_input)
        excited_input = tf.expand_dims(excited_input,1)
        excited_input = tf.expand_dims(excited_input,1)
        return excited_input * x

class SpatialSqueezeAndExcite(keras.layers.Layer):
    def __init__(self):
        self.setup_layers()
    
    def setup_layers(self):
        self.layer = keras.Sequential([
            keras.layers.Conv2D(1,1,padding='same'),
            keras.layers.Activation('sigmoid')])
    
    def call(self,x):
        return self.layer(x) * x

class SCSqueezeAndExcite(keras.layers.Layer):
    def __init__(self,n_channels):
        self.n_channels = n_channels
        self.setup_layers()

    def setup_layers(self,x):
        self.spatial_sae = SpatialSqueezeAndExcite()
        self.channel_sae = ChannelSqueezeAndExcite(self.n_channels)

    def call(self,x):
        return self.spatial_sae(x) + self.channel_sae(x)

class UNet(keras.Model):
    def __init__(self,
                 depth_mult=1.,
                 padding='VALID',
                 factorization=False,
                 n_classes=2,
                 beta=0.005,
                 squeeze_and_excite=False,
                 dropout_rate=0.2,
                 loss_fn=None):
        super(UNet, self).__init__()
        self.depth_mult = depth_mult
        self.padding = padding
        self.factorization = factorization
        self.n_classes = n_classes
        self.beta = beta
        self.squeeze_and_excite = squeeze_and_excite
        self.dropout_rate = dropout_rate
        self.loss_fn = loss_fn # used in train_step

        self.depths = [64,128,256,512,1024]
        self.depths = [int(x*self.depth_mult) for x in self.depths]
        self.setup_network()

    def setup_network(self):
        self.reductions = [
            UNetConvBlock(
                depth=self.depths[0],conv_size=3,
                factorization=self.factorization,padding=self.padding,
                dropout_rate=self.dropout_rate,beta=self.beta)]
        for depth in self.depths[1:-1]:
            m = UNetReductionBlock(
                depth=depth,conv_size=3,
                factorization=self.factorization,padding=self.padding,
                dropout_rate=self.dropout_rate,beta=self.beta)
            self.reductions.append(m)

        self.bottleneck_layer_1 = UNetConvBlock(
            depth=self.depths[-2],conv_size=3,stride=1,
            factorization=self.factorization,padding=self.padding,
            dropout_rate=self.dropout_rate,beta=self.beta)
        self.bottleneck_layer_2 = UNetConvLayer(
            depth=self.depths[-1],conv_size=3,stride=2,
            factorization=self.factorization,padding=self.padding,
            dropout_rate=self.dropout_rate,beta=self.beta)

        self.reconstructions = []
        for depth in self.depths[-2::-1]:
            m = UNetReconstructionBlock(
                depth=self.depths[0],conv_size=3,
                factorization=self.factorization,padding=self.padding,
                dropout_rate=self.dropout_rate,beta=self.beta)
            self.reconstructions.append(m)
        
        self.classification_layer = keras.layers.Conv2D(
            self.n_classes,1)
    
    def call(self,x):
        red_1 = self.reductions[0](x)
        red_2,pre_1 = self.reductions[1](red_1)
        red_3,pre_2 = self.reductions[2](red_2)
        red_4,pre_3 = self.reductions[3](red_3)

        pre_4 = self.bottleneck_layer_1(red_4)
        red_5 = self.bottleneck_layer_2(pre_4)
        
        rec_1 = self.reconstructions[0](red_5,pre_4)
        rec_2 = self.reconstructions[1](rec_1,pre_3)
        rec_3 = self.reconstructions[2](rec_2,pre_2)
        rec_4 = self.reconstructions[3](rec_3,pre_1)

        classification = self.classification_layer(rec_4)

        return classification

    def train_step(self, data):
        x, y, w = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(
                y,y_pred,w,regularization_losses=self.losses,)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

class WeightedCrossEntropy(keras.losses.Loss):
    def __init__(self):
        super(WeightedCrossEntropy,self).__init__()
        self.loss = keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)

    def call(self,y_true,y_pred,sample_weight,regularization_losses=None):
        if tf.rank(w) == 4:
            w = w[:,:,:,0]
        l = self.loss(y_true,y_pred)
        l = l * w
        l = tf.reduce_mean(l,axis=[1,2])
        if model is not None:
            l = l + tf.add_n(model.losses) / len(model.losses)
        return l
    def __call__(self,y_true,y_pred,sample_weight,regularization_losses=None):
        return self.call(
            self,y_true,y_pred,sample_weight,regularization_losses=None)

class TrainUpdater:
    def __init__(self,optimizer,loss):
        self.optimizer = optimizer
        self.loss = loss
        self.loss_average = keras.metrics.Mean()
        self.current_input = None
        self.current_y = None
        self.current_y_pred = None
        self.current_w = None

    def __call__(self,model,x,y,w):
        self.update(model,x,y,w)

    def update(self,model,x,y,w):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            l = self.loss(y,y_pred,w,model)

        trainable_vars = model.trainable_variables
        gradients = tape.gradient(l, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.current_input = x
        self.current_y = y
        self.current_y_pred = y_pred
        self.current_w = w
        self.current_l = l
        self.loss_average.update_state(l)
    
    def reset(self):
        self.loss_average.reset_states()
    
    def get_loss(self):
        return self.loss_average.result()

@tf.function
def weighted_binary_cross_entropy(y_true,y_pred,sample_weight):
    l = tf.keras.metrics.binary_crossentropy(y_true,y_pred)
    l = l*sample_weight
    return tf.reduce_mean(l,axis=[1,2])

class HDF5Dataset:
    def __init__(self,h5py_path,
                 input_height=256,input_width=256,
                 key_list=None,augment_fn=None):
        self.h5py_path = h5py_path
        self.input_height = input_height
        self.input_width = input_width
        self.key_list = key_list
        self.augment_fn = augment_fn
        self.segmentation_dataset = SegmentationDataset(
            hdf5_file=self.h5py_path,
            dimensions=(0,0,self.input_height,self.input_width),
            mode='segmentation',
            rel_keys=['image','mask','weight_map'],
            rotate_record=True,
            transform=None)
        self.size = len(self.segmentation_dataset)
        self.key_list = [x for x in self.key_list 
                         if x in self.segmentation_dataset.hf_keys]
    
    def grab(self,augment=True):
        if self.key_list is None:
            random_idx = np.random.randint(0,self.size)
            rr = self.segmentation_dataset[random_idx]
            mask = np.concatenate([1-rr['mask'],rr['mask']],axis=2)
        else:
            random_key = np.random.choice(self.key_list)
            rr = self.segmentation_dataset[random_key]
            mask = np.concatenate([1-rr['mask'],rr['mask']],axis=2)
        image = tf.convert_to_tensor(rr['image'])
        image = tf.cast(image,tf.float32) / 255.
        mask = tf.convert_to_tensor(mask * 255)
        mask = tf.cast(mask,tf.float32)
        if tf.reduce_any(mask == 255.):
            mask = mask / 255.
        weight_map = tf.convert_to_tensor(rr['weight_map'])
        if augment == True:
            if self.augment_fn is not None:
                image,mask,weight_map = self.augment_fn(image,mask,weight_map)
        return image,mask,weight_map

def generate_images_h5py_dataset(h5py_path,
                                input_height=256,
                                input_width=256,
                                key_list=None,
                                augment_fn=None):
    segmentation_dataset = SegmentationDataset(
        hdf5_file=h5py_path,
        dimensions=(0,0,input_height,input_width),
        mode='segmentation',
        rel_keys=['image','mask','weight_map'],
        rotate_record=True,
        transform=None)
    size = len(segmentation_dataset)
    key_list = [x for x in key_list if x in segmentation_dataset.hf_keys]
    while True:
        if key_list is None:
            random_idx = np.random.randint(0,size)
            rr = segmentation_dataset[random_idx]
            mask = np.concatenate([1-rr['mask'],rr['mask']],axis=2)
        else:
            random_key = np.random.choice(key_list)
            rr = segmentation_dataset[random_key]
            mask = np.concatenate([1-rr['mask'],rr['mask']],axis=2)

        image = tf.convert_to_tensor(rr['image'])
        image = tf.cast(image,tf.float32) / 255.
        mask = tf.convert_to_tensor(mask * 255)
        weight_map = tf.convert_to_tensor(rr['weight_map'])
        if augment_fn is not None:
            image,mask,weight_map = augment_fn(image,mask,weight_map)
        yield image,mask,weight_map