import argparse
import tqdm
import tensorflow as tf
from tensorflow import keras

from data_generators import *
from unet_utilities import *
from tf_da import *

def load_generator():
    return generate_images_h5py_dataset(
        h5py_path=args.dataset_path,input_height=args.input_height,
        input_width=args.input_width,key_list=key_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains U-Net model.')

    parser.add_argument('--dataset_path',dest='dataset_path',
                        action='store',type=str,default=None)
    parser.add_argument('--input_height',dest = 'input_height',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--input_width',dest = 'input_width',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')

    parser.add_argument('--save_summary_folder',dest = 'save_summary_folder',
                        action = 'store',type = str,default = 'summaries',
                        help = 'Directory where summaries are saved.')
    parser.add_argument('--batch_size',dest = 'batch_size',
                        action = 'store',type = int,default = 4,
                        help = 'Size of mini batch.')
    parser.add_argument('--truth_only',dest = 'truth_only',
                        action = 'store_true',default = False,
                        help = 'Consider only images with at least one class.')

    #Data augmentation
    for arg in [
        ['brightness_max_delta',8. / 255.,float],
        ['saturation_lower',0.1,float],
        ['saturation_upper',1.1,float],
        ['hue_max_delta',0.1,float],
        ['contrast_lower',0.1,float],
        ['contrast_upper',1.1,float],
        ['noise_stddev',0.01,float],
        ['blur_probability',0.1,float],
        ['blur_size',3,int],
        ['blur_mean',0,float],
        ['blur_std',0.05,float],
        ['discrete_rotation',True,'store_true'],
        ['min_jpeg_quality',100,int],
        ['max_jpeg_quality',100,int],
        ['elastic_transform_p',0.1,float]]:
        if arg[2] != 'store_true':
            parser.add_argument('--{}'.format(arg[0]),dest=arg[0],
                                action='store',type=arg[2],default=arg[1])
        else:
            parser.add_argument('--{}'.format(arg[0]),dest=arg[0],
                                action='store_true',default=False)
    

    parser.add_argument('--noise_chance',dest = 'noise_chance',
                        action = 'store',type = float,
                        default = 0.1,
                        help = 'Probability to add noise.')
    parser.add_argument('--blur_chance',dest = 'blur_chance',
                        action = 'store',type = float,
                        default = 0.05,
                        help = 'Probability to blur the input image.')
    parser.add_argument('--key_list',dest = 'key_list',
                        action = 'store',
                        default = None,
                        help = 'File with one image file per list (for h5 \
                        extension).')

    args = parser.parse_args()

    data_augmentation_params = {
        'brightness_max_delta':args.brightness_max_delta,
        'saturation_lower':args.saturation_lower,
        'saturation_upper':args.saturation_upper,
        'hue_max_delta':args.hue_max_delta,
        'contrast_lower':args.contrast_lower,
        'contrast_upper':args.contrast_upper,
        'noise_stddev':args.noise_stddev,
        'blur_probability':args.blur_probability,
        'blur_size':args.blur_size,
        'blur_mean':args.blur_mean,
        'blur_std':args.blur_std,
        'discrete_rotation':args.discrete_rotation,
        'min_jpeg_quality':args.min_jpeg_quality,
        'max_jpeg_quality':args.max_jpeg_quality,
        'elastic_transform_p':args.elastic_transform_p
    }

    print("Setting up data generator...")
    IA = ImageAugmenter(**data_augmentation_params)
    key_list = [x.strip() for x in open(args.key_list).readlines()]
    def load_generator():
        return generate_images_h5py_dataset(
            h5py_path='../u-net/segmentation_dataset.h5',
            input_height=args.input_height,input_width=args.input_width,key_list=key_list,
            augment_fn=IA.augment)
    generator = load_generator
    output_types = (tf.float32,tf.float32,tf.float32)
    output_shapes = (
        tf.TensorShape((args.input_height,args.input_width,3)),
        tf.TensorShape((args.input_height,args.input_width,2)),
        tf.TensorShape((args.input_height,args.input_width,1)))
    tf_dataset = tf.data.Dataset.from_generator(
        generator,output_types=output_types,output_shapes=output_shapes)
    if args.truth_only == True:
        tf_dataset = tf_dataset.filter(
            lambda x,y,w: tf.reduce_sum(y[:,:,1:]) > 0.)
    tf_dataset = tf_dataset.batch(args.batch_size)
    tf_dataset = tf_dataset.prefetch(50)
    
    print("Training...")
    writer = tf.summary.create_file_writer(args.save_summary_folder)
    tf_dataset_iterable = iter(tf_dataset)
    with writer.as_default():
        for i in range(10000):
            x,y,w = next(tf_dataset_iterable)
            tf.summary.image(
                "InputImage", x, step=i)
            tf.summary.image(
                "GroundTruth", y, step=i)
            tf.summary.image(
                "WeightMap", tf.expand_dims(w,axis=-1), step=i)

            x = x.numpy()
            y = y.numpy()
            w = w.numpy()
            print("Range in image", np.min(x),np.max(x))
            print("Unique in labels", np.unique(y))
            print("Range in weights", np.min(w),np.max(w))
