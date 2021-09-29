from math import pi
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import cv2

from albumentations import ElasticTransform

"""
Set of functions used to perform data augmentation on tf.
"""

class ElasticTransformWrapper:
    def __init__(self,sigma,alpha,alpha_affine,p):
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.p = p

        self.et = ElasticTransform(
            sigma=self.sigma,alpha=self.alpha,
            alpha_affine=self.alpha_affine,p=self.p)

    def __call__(self,image,*masks):
        out = self.et(image=image,masks=masks)
        image,masks = out['image'],out['masks']
        out = [image,*masks]
        return out

class ImageAugmenter:
    def __init__(self,
                 brightness_max_delta=16. / 255.,
                 saturation_lower=0.8,
                 saturation_upper=1.2,
                 hue_max_delta=0.2,
                 contrast_lower=0.8,
                 contrast_upper=1.2,
                 noise_stddev=0.05,
                 blur_probability=0.1,
                 blur_size=3,
                 blur_mean=0.,
                 blur_std=0.05,
                 discrete_rotation=True,
                 min_jpeg_quality=30,
                 max_jpeg_quality=70,
                 elastic_transform_sigma=10,
                 elastic_transform_alpha_affine=0,
                 elastic_transform_p=0.7):

        self.brightness_max_delta = brightness_max_delta
        self.saturation_lower = saturation_lower
        self.saturation_upper = saturation_upper
        self.hue_max_delta = hue_max_delta
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.noise_stddev = noise_stddev
        self.blur_probability = blur_probability
        self.blur_size = blur_size
        self.blur_mean = blur_mean
        self.blur_std = blur_std
        self.discrete_rotation = discrete_rotation
        self.min_jpeg_quality = min_jpeg_quality
        self.max_jpeg_quality = max_jpeg_quality
        self.elastic_transform_sigma = elastic_transform_sigma
        self.elastic_transform_alpha_affine = elastic_transform_alpha_affine
        self.elastic_transform_p = elastic_transform_p

    def __str__(self):
        return "ImageAugmenter class"

    def augment(self,image,*masks):
        image = tf.image.convert_image_dtype(image,tf.float32)
        masks = [tf.image.convert_image_dtype(m,tf.float32) for m in masks]

        if self.elastic_transform_p > 0:
            image,masks = elastic_transform(
                image,*masks,
                sigma=self.elastic_transform_sigma,
                alpha_affine=self.elastic_transform_alpha_affine,
                p=self.elastic_transform_p)

        image_shape = image.shape.as_list()
        image = random_color_transformations(image,
                                             self.brightness_max_delta,
                                             self.saturation_lower,
                                             self.saturation_upper,
                                             self.hue_max_delta,
                                             self.contrast_lower,
                                             self.contrast_upper)

        image = gaussian_blur(image,
                              self.blur_probability,self.blur_size,
                              self.blur_mean,self.blur_std)

        image = gaussian_noise(image,self.noise_stddev)

        image,masks = random_rotation(
            image,*masks,
            discrete_rotation=self.discrete_rotation)

        if self.min_jpeg_quality - self.max_jpeg_quality != 0:
            image = random_jpeg_quality(image,
                                        self.min_jpeg_quality,
                                        self.max_jpeg_quality
                                        )
        image = tf.reshape(image,image_shape)
        
        if len(masks) == 0:
            return image
        else:
            return image,(*masks)

def random_color_transformations(
    image,
    brightness_max_delta,
    saturation_lower,saturation_upper,
    hue_max_delta,
    contrast_lower,contrast_upper):

    """
    Function to randomly alter an images brightness, saturation, hue and
    contrast.

    Parameters:
    * image - three channel image (H,W,3)
    * brightness_params - dictionary with parameters for
    tf.image.random_brightness
    * saturation_params - dictionary with parameters for
    tf.image.random_saturation
    * hue_params - dictionary with parameters for tf.image.random_hue
    * contrast_params - dictionary with parameters for tf.image.random_contrast
    """

    def brightness(x):
        if brightness_max_delta != 0:
            return tf.image.random_brightness(x,brightness_max_delta)
        else:
            return x

    def saturation(x):
        if saturation_lower - saturation_upper != 0:
            return tf.image.random_brightness(
                x,saturation_lower,saturation_upper)
        else:
            return x
    def hue(x):
        if hue_max_delta != 0:
            return tf.image.random_hue(x,hue_max_delta)
        else:
            return x

    def contrast(x):
        if contrast_lower - contrast_upper != 0:
            return tf.image.random_contrast(
                x,contrast_lower,contrast_upper)
        else:
            return x

    def distort_colors_0(image):
        image = brightness(image)
        image = saturation(image)
        image = hue(image)
        image = contrast(image)
        return image

    def distort_colors_1(image):
        image = saturation(image)
        image = brightness(image)
        image = contrast(image)
        image = hue(image)
        return image

    def distort_colors_2(image):
        image = contrast(image)
        image = hue(image)
        image = brightness(image)
        image = saturation(image)
        return image

    def distort_colors(image,color_ordering):
        if color_ordering == 0:
            return distort_colors_0(image)
        if color_ordering == 1:
            return distort_colors_1(image)
        if color_ordering == 2:
            return distort_colors_2(image)
        return image

    color_ordering = tf.random.uniform([],0,4,tf.int32)
    image = distort_colors(image,color_ordering)
    image = tf.clip_by_value(image,0.,1.)
    return image

def salt_and_pepper(image,salt_prob=0.01,pepper_prob=0.01):
    def get_mask(h,w,p):
        return tf.expand_dims(
            tf.where(tf.random.uniform(shape=(h,w),minval=0.,maxval=1.) > p,
            tf.ones((h,w)),
            tf.zeros((h,w))),
            axis=2)

    image_shape = tf.shape(image)
    image_shape_list = image.shape.as_list()
    salt_mask = get_mask(image_shape[0],image_shape[1],salt_prob)
    pepper_mask = get_mask(image_shape[0],image_shape[1],pepper_prob)

    image = tf.where(salt_mask == 1.,tf.ones_like(image),image)
    image = tf.where(pepper_mask == 1.,tf.zeros_like(image),image)

    return image

def gaussian_noise(image,stddev=0.01):
    image = image + tf.random.normal(
        tf.shape(image),stddev=stddev)
    image = tf.clip_by_value(image,0.,1.)
    return image

def random_rotation(image,*masks,discrete_rotation=True):
    flip_lr_prob = tf.random.uniform([]) > 0.5
    flip_ud_prob = tf.random.uniform([]) > 0.5

    if flip_lr_prob == True:
        image = tf.image.flip_left_right(image)
        masks = [tf.image.flip_left_right(m) for m in masks]
    if flip_ud_prob == True:
        image = tf.image.flip_up_down(image)
        masks = [tf.image.flip_up_down(m) for m in masks]

    if discrete_rotation == True:
        rot90_prob = tf.random.uniform([]) > 0.5
        rot90_angle = tf.random.uniform([],minval=0,maxval=4,
                                        dtype=tf.int32)
        if rot90_prob == True:
            image = tf.image.rot90(image,rot90_angle)
            masks = [tf.image.rot90(m,rot90_angle) for m in masks]

    return image, masks

def gaussian_blur(
    image,
    blur_probability=0.1,
    size=1,
    mean=0.0,
    std=0.05):
    """
    Function to randomly apply a gaussian blur on an image. 
    Based on https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow/52012658

    Parameters:
    * image - three channel image (H,W,3)
    * blur_probability - probability for bluring
    * size - kernel size
    * mean - distribution mean
    * std - distribution std
    """

    def gaussian_kernel(size,mean,std):
        """Makes 2D gaussian Kernel for convolution."""

        d = tfp.distributions.Normal(float(mean), float(std))

        vals = d.prob(tf.range(start=-size,limit=size+1,dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij',vals,vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    image_shape = image.shape.as_list()
    gaussian_filter = gaussian_kernel(size,mean,std)
    gaussian_filter = tf.stack([gaussian_filter for _ in range(3)],axis=-1)
    gaussian_filter = tf.stack([gaussian_filter for _ in range(3)],axis=-1)
    if tf.random.uniform([]) < blur_probability:
        transformed_image = tf.nn.conv2d(
            tf.expand_dims(image,axis=0),
            gaussian_filter,strides=[1,1,1,1],padding="SAME")
    transformed_image = tf.reshape(image,image_shape)
    return transformed_image

def random_jpeg_quality(image,
                        min_jpeg_quality=30,
                        max_jpeg_quality=70):
    """
    Function to randomly alter JPEG quality.

    Parameters:
    * image - three channel image (H,W,3)
    * min_jpeg_quality - minimum JPEG quality
    * max_jpeg_quality - maximum JPEG quality
    """

    return tf.image.random_jpeg_quality(image,
                                        min_jpeg_quality,
                                        max_jpeg_quality)

def elastic_transform(image,*masks,sigma=10,alpha_affine=10,p=0.7):
    """
    Applies elastic distortion (elastic transform) to images and their
    respective masks. Requires

    Parameters:
    * image - three channel image (H,W,3)
    * masks - masks to be augmented with the image
    * sigma, alpha_affine, p - parameters for the ElasticTransform class
    """

    ET = ElasticTransformWrapper(
        sigma=sigma,alpha=100,
        alpha_affine=alpha_affine,p=p
    )

    shapes = [x.shape.as_list() for x in [image,*masks]]
    types_out = [tf.float32,*[tf.float32 for _ in masks]]

    out = tf.numpy_function(ET,[image,*masks],Tout=types_out)

    out = [tf.reshape(out[i],shapes[i]) for i in range(len(out))]

    image = out[0]
    masks = out[1:]

    return image, masks
