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
                y,y_pred,w,regularization_losses=self.losses)

        # ensures loss in metrics is also updated
        self.compiled_loss(
            y,y_pred,sample_weight=None,regularization_losses=self.losses) 

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

class WeightedCrossEntropy(keras.losses.Loss):
    def __init__(self):
        super(WeightedCrossEntropy,self).__init__()
        self.loss = keras.losses.CategoricalCrossentropy(
            from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

    def call(self,y_true,y_pred,w,regularization_losses=None):
        l = self.loss(y_true,y_pred)
        l = l * w[:,:,:,0]
        l = tf.reduce_mean(l,axis=[1,2])
        if regularization_losses is not None:
            l = l + tf.add_n(regularization_losses)/len(regularization_losses)
        return l

    def __call__(self,y_true,y_pred,w,regularization_losses=None):
        return self.call(y_true,y_pred,w,regularization_losses=None)

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

class HDF5Dataset:
    def __init__(self,h5py_path,
                 input_height=256,input_width=256,
                 key_list=None,augment_fn=None,
                 excluded_key_list=None):
        self.h5py_path = h5py_path
        self.input_height = input_height
        self.input_width = input_width
        self.key_list = key_list
        self.augment_fn = augment_fn
        self.excluded_key_list = excluded_key_list
        self.segmentation_dataset = SegmentationDataset(
            hdf5_file=self.h5py_path,
            dimensions=(0,0,self.input_height,self.input_width),
            mode='segmentation',
            rel_keys=['image','mask','weight_map'],
            rotate_record=True,
            transform=None)
        self.average_image_side = np.mean(
            [x[0] for x in self.segmentation_dataset.sizes])
        self.size = len(self.segmentation_dataset)
        if self.key_list is not None:
            self.key_list = [
                x for x in self.key_list 
                if x in self.segmentation_dataset.hf_keys]
        else:
            self.key_list = [
                x for x in self.segmentation_dataset.hf_keys]
        if excluded_key_list is not None:
            self.key_list = [
                x for x in self.key_list 
                if x not in self.excluded_key_list]
    
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

class HDF5DatasetTest:
    def __init__(self,h5py_path,
                 key_list=None,
                 excluded_key_list=None):
        self.h5py_path = h5py_path
        self.key_list = key_list
        self.excluded_key_list = excluded_key_list
        self.segmentation_dataset = h5py.File(self.h5py_path,'r')
        self.size = len(self.segmentation_dataset)
        if self.key_list is not None:
            K = list(self.segmentation_dataset.keys())
            self.key_list = [
                x for x in self.key_list 
                if x in K]
        else:
            self.key_list = [
                x for x in self.segmentation_dataset.keys()]
        if excluded_key_list is not None:
            self.key_list = [
                x for x in self.key_list 
                if x not in self.excluded_key_list]
    
    def generate(self,augment=True):
        for key in self.key_list:
            rr = self.segmentation_dataset[key]
            image = rr['image'][()]
            image = tf.convert_to_tensor(image)
            image = tf.cast(image,tf.float32)
            if np.any(image == 255):
                image = image / 255.
            mask = tf.convert_to_tensor(rr['mask'][()])
            mask = tf.cast(mask,tf.float32)
            mask = tf.concat([1-mask,mask],axis=-1)
            if np.any(np.isnan(image)) or np.any(np.isnan(mask)):
                pass
            else:   
                yield image,mask

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

class ImageCallBack(keras.callbacks.Callback):
    # writes images to summary
    def __init__(self,save_every_n,tf_dataset,log_dir):
        super(ImageCallBack, self).__init__()
        self.save_every_n = save_every_n
        self.tf_dataset = tf_dataset
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.count = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.count % self.save_every_n == 0:
            batch = next(iter(self.tf_dataset.take(1)))
            x,y,w = batch
            prediction = self.model.predict(x)[:,:,:,1:]
            pred_bin = tf.expand_dims(tf.argmax(prediction,-1),axis=-1)
            truth_bin = tf.expand_dims(tf.argmax(y,-1),axis=-1)
            with self.writer.as_default():
                tf.summary.image("0:InputImage",x,self.count)
                tf.summary.image("1:GroundTruth",truth_bin,self.count)
                tf.summary.image("2:Prediction",prediction,self.count)
                tf.summary.image("3:Prediction",pred_bin,self.count)
                tf.summary.image("4:WeightMap",w,self.count)
                tf.summary.scalar("Loss",logs['loss'],self.count)
                tf.summary.scalar("MeanIoU",logs['mean_io_u'],self.count)
                tf.summary.scalar("AUC",logs['auc'],self.count)
                tf.summary.scalar("Precision",logs['precision'],self.count)
        self.count += 1

class MeanIoU(keras.metrics.MeanIoU):
    # adapts MeanIoU to work with model.fit using logits
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true,y_pred,sample_weight)

class Precision(tf.keras.metrics.Precision):
    # adapts Precision to work with model.fit using logits
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true,y_pred,sample_weight)

class AUC(tf.keras.metrics.AUC):
    # adapts AUC to work with model.fit using logits.
    # assumes only two labels are present
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)[:,:,:,1]
        return super().update_state(y_true,y_pred,sample_weight)

class DataGenerator:
    def __init__(self,image_folder_path):
        self.image_folder_path = image_folder_path
        self.image_paths = glob('{}/*'.format(self.image_folder_path))
        self.n_images = len(self.image_paths)
    
    def generate(self,with_path=False):
        image_idx = [x for x in range(len(self.image_paths))]
        for idx in image_idx:
            P = self.image_paths[idx]
            x = np.array(Image.open(P))[:,:,:3]
            x = tf.convert_to_tensor(x) / 255
            yield x,P

class LargeImage:
    def __init__(self,image,tile_size=[512,512],
                 output_channels=3,offset=0):
        """
        Class facilitating the prediction for large images by 
        performing all the necessary operations - tiling and 
        reconstructing the output.
        """
        self.image = image
        self.tile_size = tile_size
        self.output_channels = output_channels
        self.offset = offset
        self.h = self.tile_size[0]
        self.w = self.tile_size[1]
        self.sh = self.image.shape[:2]
        self.output = np.zeros([self.sh[0],self.sh[1],self.output_channels])
        self.denominator = np.zeros([self.sh[0],self.sh[1],1])

    def tile_image(self):
        for x in range(0,self.sh[0],self.h-self.offset):
            if x + self.tile_size[0] > self.sh[0]:
                x = self.sh[0] - self.tile_size[0]
            for y in range(0,self.sh[1],self.w-self.offset):
                if y + self.tile_size[1] > self.sh[1]:
                    y = self.sh[1] - self.tile_size[1]
                x_1,x_2 = x, x+self.h
                y_1,y_2 = y, y+self.w
                yield self.image[x_1:x_2,y_1:y_2,:],((x_1,x_2),(y_1,y_2))

    def update_output(self,image,coords):
        (x_1,x_2),(y_1,y_2) = coords
        self.output[x_1:x_2,y_1:y_2,:] += image
        self.denominator[x_1:x_2,y_1:y_2,:] += 1

    def return_output(self):
        return self.output/self.denominator