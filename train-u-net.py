import argparse
import tqdm
import tensorflow as tf
from tensorflow import keras

from data_generators import *
from unet_utilities import *
from tf_da import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains U-Net model.')

    parser.add_argument('--dataset_path',dest='dataset_path',
                        action='store',type=str,default=None)
    parser.add_argument('--padding',dest = 'padding',
                        action = 'store',default = 'VALID',
                        help = 'Define padding.',
                        choices = ['VALID','SAME'])
    parser.add_argument('--input_height',dest = 'input_height',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--input_width',dest = 'input_width',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')

    parser.add_argument('--log_every_n_steps',dest = 'log_every_n_steps',
                        action = 'store',type = int,default = 100,
                        help = 'How often are the loss and global step logged.')
    parser.add_argument('--save_summary_folder',dest = 'save_summary_folder',
                        action = 'store',type = str,default = 'summaries',
                        help = 'Directory where summaries are saved.')
    parser.add_argument('--save_checkpoint_steps',dest = 'save_checkpoint_steps',
                        action = 'store',type = int,default = 100,
                        help = 'How often checkpoints are saved.')
    parser.add_argument('--save_checkpoint_folder',dest = 'save_checkpoint_folder',
                        action = 'store',type = str,default = 'checkpoints',
                        help = 'Directory where checkpoints are saved.')
    parser.add_argument('--squeeze_and_excite',dest='squeeze_and_excite',
                        action='store_true',default=False,
                        help='Adds SC SqAndEx layers to the enc/dec.')
    parser.add_argument('--batch_size',dest = 'batch_size',
                        action = 'store',type = int,default = 4,
                        help = 'Size of mini batch.')
    parser.add_argument('--number_of_epochs',dest = 'number_of_epochs',
                        action = 'store',type = int,default = 50,
                        help = 'Number of training epochs.')
    parser.add_argument('--beta_l2_regularization',dest = 'beta_l2_regularization',
                        action = 'store',type = float,default = 0,
                        help = 'Beta parameter for L2 regularization.')
    parser.add_argument('--learning_rate',dest = 'learning_rate',
                        action = 'store',type = float,default = 0.001,
                        help = 'Learning rate for the SGD optimizer.')
    parser.add_argument('--factorization',dest = 'factorization',
                        action = 'store_true',default = False,
                        help = 'Use convolutional layer factorization.')
    parser.add_argument('--weighted',dest = 'weighted',
                        action = 'store_true',default = False,
                        help = 'Calculates weighted cross entropy.')
    parser.add_argument('--depth_mult',dest = 'depth_mult',
                        action = 'store',type = float,default = 1.,
                        help = 'Change the number of channels in all layers.')
    parser.add_argument('--truth_only',dest = 'truth_only',
                        action = 'store_true',default = False,
                        help = 'Consider only images with at least one class.')
    parser.add_argument('--validation_iterations',dest = 'validation_iterations',
                        action = 'store',type=int,default = 1,
                        help = 'Number of iterations for validation.')

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
    parser.add_argument('--n_classes',dest = 'n_classes',
                        action = 'store',type = int,
                        default = 2,
                        help = 'Number of classes in the segmented images.')
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

    print("Setting up network...")
    u_net = UNet(depth_mult=args.depth_mult,padding=args.padding,
                 factorization=args.factorization,n_classes=args.n_classes,
                 beta=args.beta_l2_regularization,dropout_rate=0.25,
                 squeeze_and_excite=args.squeeze_and_excite)

    print("Setting up data generator...")
    IA = ImageAugmenter(**data_augmentation_params)
    key_list = [x.strip() for x in open(args.key_list).readlines()]
    hdf5_dataset = HDF5Dataset(
        h5py_path=args.dataset_path,
        input_height=args.input_height,input_width=args.input_width,
        key_list=key_list,
        augment_fn=IA.augment)
    
    def load_generator():
        while True:
            yield hdf5_dataset.grab()
    def load_generator_no_transform():
        while True:
            yield hdf5_dataset.grab(augment=False)

    generator = load_generator
    output_types = (tf.float32,tf.float32,tf.float32)
    output_shapes = (
        tf.TensorShape((args.input_height,args.input_width,3)),
        tf.TensorShape((args.input_height,args.input_width,args.n_classes)),
        tf.TensorShape((args.input_height,args.input_width,1)))
    tf_dataset = tf.data.Dataset.from_generator(
        load_generator,output_types=output_types,
        output_shapes=output_shapes)
    tf_dataset_val = tf.data.Dataset.from_generator(
        load_generator_no_transform,output_types=output_types,
        output_shapes=output_shapes)
    if args.truth_only == True:
        tf_dataset = tf_dataset.filter(
            lambda x,y,w: tf.reduce_sum(y[:,:,1:]) > 0.)
        tf_dataset_val = tf_dataset_val.filter(
            lambda x,y,w: tf.reduce_sum(y[:,:,1:]) > 0.)
    tf_dataset = tf_dataset.batch(args.batch_size).prefetch(100)
    tf_dataset_val = tf_dataset_val.batch(args.batch_size).prefetch(100)

    print("Setting up training...")
    iou = MeanIoU(args.n_classes)
    auc = AUC()
    prec = Precision()
    loss_fn = WeightedCrossEntropy()
    loss_fn_compile = keras.losses.CategoricalCrossentropy(from_logits=True)
    u_net.compile(
        optimizer=keras.optimizers.Adamax(learning_rate=args.learning_rate),
        loss=loss_fn_compile, metrics=[iou,auc,prec])
    u_net.loss_fn = loss_fn
    steps_per_epoch = hdf5_dataset.size // args.batch_size
    steps_per_epoch *= hdf5_dataset.average_image_side / args.input_height
    steps_per_epoch = int(steps_per_epoch)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=args.save_summary_folder)
    image_callback = ImageCallBack(
        args.log_every_n_steps,tf_dataset,log_dir=args.save_summary_folder)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        args.save_checkpoint_folder + '-{epoch:02d}')
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        'val_loss',min_lr=1e-6,factor=0.75)

    all_callbacks = [
        tensorboard_callback,image_callback,checkpoint_callback,
        lr_callback]

    print("Training...")
    u_net.fit(
        x=tf_dataset,batch_size=args.batch_size,epochs=args.number_of_epochs,
        callbacks=all_callbacks,steps_per_epoch=steps_per_epoch,
        validation_data=tf_dataset_val,validation_steps=15)