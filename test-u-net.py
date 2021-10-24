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

    parser.add_argument('--checkpoint_path',dest = 'checkpoint_path',
                        action = 'store',type = str,default = 'checkpoints',
                        help = 'Path to checkpoint to restore.')
    parser.add_argument('--squeeze_and_excite',dest='squeeze_and_excite',
                        action='store_true',default=False,
                        help='Adds SC SqAndEx layers to the enc/dec.')
    parser.add_argument('--factorization',dest = 'factorization',
                        action = 'store_true',default = False,
                        help = 'Use convolutional layer factorization.')
    parser.add_argument('--depth_mult',dest = 'depth_mult',
                        action = 'store',type = float,default = 1.,
                        help = 'Change the number of channels in all layers.')

    parser.add_argument('--n_classes',dest = 'n_classes',
                        action = 'store',type = int,
                        default = 2,
                        help = 'Number of classes in the segmented images.')
    parser.add_argument('--tta',dest = 'tta',action = 'store_true',
                        help = 'Use test-time augmentation.')
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

    args = parser.parse_args()

    print("Setting up network...")
    u_net = UNet(depth_mult=args.depth_mult,padding=args.padding,
                 factorization=args.factorization,n_classes=args.n_classes,
                 dropout_rate=0,squeeze_and_excite=args.squeeze_and_excite)
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

    hdf5_dataset = HDF5DatasetTest(
        h5py_path=args.dataset_path,
        key_list=key_list,
        excluded_key_list=excluded_key_list)

    print("Setting up training...")
    iou = MeanIoU(args.n_classes)
    auc = AUC()
    prec = Precision()

    for image,mask in hdf5_dataset.generate():
        large_image = LargeImage(image,[512,512],args.n_classes,offset=128)
        for tile,coords in large_image.tile_image():
            tile_tensor = tf.convert_to_tensor(tile)
            tile_tensor = tf.expand_dims(tile_tensor,0)
            if args.tta == True:
                tile_tensor = tta_rotation(tile_tensor)
                prediction = u_net(tile_tensor)
                prediction = rotate_and_reduce(prediction)
                prediction = np.squeeze(prediction.numpy(),axis=0)
            else:
                prediction = u_net(tile_tensor)
                prediction = np.squeeze(prediction.numpy(),axis=0)
            large_image.update_output(prediction,coords)

        mask = tf.convert_to_tensor(mask)
        prediction = tf.convert_to_tensor(large_image.return_output())
        mask = tf.expand_dims(mask,0)
        prediction = tf.expand_dims(prediction,0)
        auc.update_state(mask,prediction)
        iou.update_state(mask,prediction)
        prec.update_state(mask,prediction)
        
    print("TEST,IOU,global,{}".format(float(iou.result().numpy())))
    print("TEST,AUC,global,{}".format(float(auc.result().numpy())))
    print("TEST,Precision,global,{}".format(float(prec.result().numpy())))