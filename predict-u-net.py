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
    parser.add_argument('--key_list',dest = 'key_list',
                        action = 'store',
                        default = None,
                        help = 'File with one image file per list (for h5 \
                        extension).')
    parser.add_argument('--excluded_key_list',dest = 'excluded_key_list',
                        action = 'store',
                        default = None,
                        help = 'File with one image file per list (for h5 \
                        extension) to be excluded.')

    parser.add_argument('--input_height',dest = 'input_height',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--input_width',dest = 'input_width',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--padding',dest = 'padding',
                        action = 'store',default = 'VALID',
                        help = 'Define padding.',
                        choices = ['VALID','SAME'])
    parser.add_argument('--depth_mult',dest = 'depth_mult',
                        action = 'store',type = float,default = 1.,
                        help = 'Change the number of channels in all layers.')
    parser.add_argument('--rs',dest = 'rs',action = 'store',
                        type=float,default=1,
                        help = 'Rescale by this factor (only if refining).')

    parser.add_argument('--checkpoint_path',dest = 'checkpoint_path',
                        action = 'store',type = str,default = 'checkpoints',
                        help = 'Path to checkpoint.')

    parser.add_argument('--n_classes',dest = 'n_classes',
                        action = 'store',type = int,
                        default = 2,
                        help = 'Number of classes in the segmented images.')

    args = parser.parse_args()

    print("Setting up network...")
    u_net = UNet(depth_mult=args.depth_mult,padding=args.padding,
                 factorization=False,n_classes=args.n_classes,
                 dropout_rate=0,squeeze_and_excite=False)
    u_net.load_weights(args.checkpoint_path)
    u_net.training = False

    print("Setting up data generator...")
    if args.key_list is not None:
        key_list = [x.strip() for x in open(args.key_list).readlines()]
    else:
        key_list = None
    if args.excluded_key_list is not None:
        excluded_key_list = [
            x.strip() for x in open(args.excluded_key_list).readlines()]
    else:
        excluded_key_list = None
    hdf5_dataset = HDF5DatasetPredict(
        h5py_path=args.input_path,
        key_list=key_list,
        excluded_key_list=excluded_key_list)
    hdf5_output = h5py.File(args.output_path,'w')

    print("Predicting...")
    for image,path in tqdm(hdf5_dataset.generate(with_path=True)):
        R = os.path.split(path)[-1]
        g = hdf5_output.create_group(R)
        g['image'] = image
        large_image = LargeImage(image,[512,512],args.n_classes,offset=128)
        large_image_tta = LargeImage(image,[512,512],args.n_classes,offset=128)
        for tile,coords in large_image.tile_image():
            tile_tensor = tf.convert_to_tensor(tile)
            tile_tensor = tf.expand_dims(tile_tensor,0)
            tile_tensor = tta_rotation(tile_tensor)
            prediction = u_net(tile_tensor)
            prediction_original = prediction[0,:,:,:]
            prediction = rotate_and_reduce(prediction)
            prediction = np.squeeze(prediction.numpy(),axis=0)
            prediction_original = prediction_original.numpy()
            large_image.update_output(prediction_original,coords)
            large_image_tta.update_output(prediction,coords)

        pred_np = large_image.return_output()
        pred_np_tta = large_image_tta.return_output()
        g['prediction'] = pred_np
        g['prediction_tta'] = pred_np_tta

        pred_tta = tf.convert_to_tensor(pred_np_tta)
        pred_np_tta = tf.keras.activations.softmax(
            pred_tta,axis=-1)[:,:,1]
        pred_np_tta = pred_np_tta.numpy()
        pred_np_tta_refine = refine_prediction_wbc(image,pred_np_tta,args.rs)

        g['prediction_tta_refine'] = pred_np_tta_refine