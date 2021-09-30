import argparse
from tqdm import tqdm
from skimage.color import label2rgb
import tensorflow as tf
from tensorflow import keras

from unet_utilities import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predicts using U-Net model.')

    parser.add_argument('--input_path',dest='input_path',
                        action='store',type=str,default=None)
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None)
    parser.add_argument(
        '--example',dest = 'example',
        action = 'store_true',
        help = 'Combines input and prediction when saving the output.')
    parser.add_argument('--tta',dest = 'tta',action = 'store_true',
                        help = 'User test-time augmentation.')

    parser.add_argument('--input_height',dest = 'input_height',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--input_width',dest = 'input_width',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')

    parser.add_argument('--checkpoint_path',dest = 'checkpoint_path',
                        action = 'store',type = str,default = 'checkpoints',
                        help = 'Path to checkpoint.')

    parser.add_argument('--n_classes',dest = 'n_classes',
                        action = 'store',type = int,
                        default = 2,
                        help = 'Number of classes in the segmented images.')

    args = parser.parse_args()

    try: os.makedirs(args.output_path)
    except: pass

    u_net = keras.models.load_model(args.checkpoint_path)

    data_generator = DataGenerator(args.input_path)

    pbar = tqdm(data_generator.n_images)
    for image,image_path in data_generator.generate(with_path=True):
        root = os.path.split(image_path)[-1]
        output_path = '{}/prediction_{}'.format(args.output_path,root)
        condition_h = image.shape[0] <= args.input_height
        condition_w = image.shape[1] <= args.input_width
        if np.all([condition_h,condition_w]):
            image_tensor = tf.expand_dims(
                tf.convert_to_tensor(image),axis=0)
            image_prediction = u_net.predict(image_tensor)[0]
        else:
            large_image = LargeImage(
                image,tile_size=[args.input_height,args.input_width],
                output_channels=args.n_classes)
            for tile,coords in large_image.tile_image():
                tile_tensor = tf.expand_dims(
                    tf.convert_to_tensor(tile),axis=0)
                tile_prediction = u_net.predict(tile_tensor)
                large_image.update_output(tile_prediction[0],coords)
            image_prediction = large_image.return_output()
        image_prediction = np.argmax(image_prediction,axis=-1)

        image_prediction = label2rgb(image_prediction,bg_label=0)
        if args.example == True:
            image_prediction = np.concatenate([image,image_prediction],axis=1)
        image_prediction = Image.fromarray(np.uint8(image_prediction*255))
        image_prediction.save(output_path)
        pbar.update()