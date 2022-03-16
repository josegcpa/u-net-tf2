import h5py
import sys
from glob import glob
import os
import numpy as np
import cv2
from math import inf
from skimage import io
from scipy.spatial import distance

def read_image(path):
    return io.imread(path)

def get_weights(truth_image,w0=0.5,sigma=10):
    """
    Has to be one channel only.

    Arguments [default]:
    * truth_image - numpy array with ground truth image
    * w0 - w0 value for the weight map [10]
    * sigma - sigma value for the weight map [5]
    """
    stride = 128
    size = 256
    sh = truth_image.shape
    max_x = sh[0] // stride
    max_y = sh[1] // stride
    if max_x % stride != 0: max_x += 1
    if max_y % stride != 0: max_y += 1
    kernel = np.ones((3,3))
    truth_image = truth_image.copy()
    truth_image[truth_image > 0] = 255
    truth_image = cv2.morphologyEx(truth_image.astype('uint8'),
                                   cv2.MORPH_OPEN, kernel)
    truth_image_ = truth_image.copy()
    edges = cv2.Canny(truth_image,100,200)

    if 255 in truth_image:
        final_weight_mask = np.zeros((sh[0],sh[1]),dtype = 'float32')
        for i in range(max_x):
            for j in range(max_y):
                m_x,M_x = i*stride,i*stride + size
                m_y,M_y = j*stride,j*stride + size
                sub_image = truth_image[m_x:M_x,m_y:M_y]
                sub_edges = edges[m_x:M_x,m_y:M_y]
                tmp_weight_mask = final_weight_mask[m_x:M_x,m_y:M_y]

                ssh = sub_image.shape
                pixel_coords = np.where(sub_image == 0)
                pixel_coords_t = np.transpose(pixel_coords)
                weight_mask = np.zeros((ssh[0],ssh[1]),dtype = 'float32')

                cp_coords = np.transpose(np.where(sub_edges > 0))
                distances = distance.cdist(pixel_coords_t,cp_coords)
                distances[distances == 0] = inf
                if distances.any():
                    mins = np.array(np.min(distances,axis = 1))
                    weight_map = ((mins * 2) ** 2)/(2 * sigma ** 2)
                    weight_map = w0 + ((1 - w0) * np.exp(-weight_map))

                    weight_mask[pixel_coords[0],
                                pixel_coords[1]] = weight_map

                else:
                    weight_mask = np.ones((ssh[0],ssh[1]),
                                          dtype = 'float32') * w0

                final_weight_mask[m_x:M_x,m_y:M_y] = np.where(
                    weight_mask > tmp_weight_mask,
                    weight_mask,
                    tmp_weight_mask
                )

    else:
        final_weight_mask = np.ones(truth_image.shape) * w0

    final_weight_mask[truth_image > 0] = 1
    return final_weight_mask

images_path = sys.argv[1]
masks_path = sys.argv[2]
all_images = glob('{}/*'.format(images_path))
image_pairs = {}
for image_path in all_images:
    mask_path = '{}/{}'.format(
        masks_path,image_path.split(os.sep)[-1])
    if os.path.isfile(mask_path):
        image_pairs[image_path] = mask_path

with h5py.File(sys.argv[3], 'w') as hf:

    for image_path in list(image_pairs.keys()):
        print(image_path)
        image = read_image(image_path)[:,:,:3]
        mask = read_image(image_pairs[image_path])[:,:,0][:,:,np.newaxis]
        mask = np.where(mask > 0,1,0).astype(np.uint8)
        edge_image = np.uint8(cv2.Laplacian(mask,cv2.CV_64F))
        num_labels, labels_im = cv2.connectedComponents(mask)
        bounding_boxes = []
        centers = []

        g = hf.create_group(image_path.split(os.sep)[-1])
        edge_group = g.create_group('edges')

        for c in range(1,num_labels):
            edge_x,edge_y = np.where((labels_im == c) & (edge_image > 0))
            object_edges = np.stack([edge_y,edge_x])
            edge_group.create_dataset(str(c),
                                      shape=object_edges.shape,
                                      dtype=np.int32,
                                      data=object_edges)
            x,y = np.where(labels_im == c)
            if len(x) < 10:
                print('\tbigmad')
            bbox = [np.min(x),np.min(y),
                    np.max(x),np.max(y)]
            bbox_center = [int((bbox[0] + bbox[2])/2),
                           int((bbox[1] + bbox[3])/2)]
            center_x,center_y = bbox_center
            bounding_boxes.append(bbox)

            centers.append([center_y,center_x])

        bounding_boxes = np.array(bounding_boxes)
        centers = np.array(centers)
        weight_map = get_weights(mask)[:,:,np.newaxis]

        g.create_dataset('image', shape=image.shape, dtype=np.uint8, data=image)
        g.create_dataset('mask', shape=mask.shape, dtype=np.uint8, data=mask)
        g.create_dataset('bounding_boxes',shape=bounding_boxes.shape,
                         dtype=np.int16,data=bounding_boxes)
        g.create_dataset('weight_map',shape=weight_map.shape,
                         dtype=np.float32,data=weight_map)
        g.create_dataset('centers',shape=centers.shape,
                         dtype=np.int32,data=np.int16(centers))
