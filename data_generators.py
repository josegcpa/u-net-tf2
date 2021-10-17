from glob import glob
import os
import h5py
import numpy as np
from scipy.spatial import distance
from math import inf

from skimage.transform import rotate
from skimage import io
from PIL import Image
import cv2

def read_image(path):
    return io.imread(path)

def show_image_mask(image,mask=None):
    if mask is not None:
        image = image * np.where(mask > 0,1,0.3)
        image = image.astype(np.uint8)
    return image

def rotate_coords(x,y,angle):
    angle = np.radians(angle)
    new_x = x*np.cos(angle) - y*np.sin(angle)
    new_y = y*np.cos(angle) + x*np.sin(angle)
    return new_x,new_y

def select_region(record,dimensions):

    x1,y1,x2,y2 = [int(x) for x in dimensions]

    if record['centers'].shape[0] > 0:
        conditions = np.array([
            record['centers'][:,0] > y1,
            record['centers'][:,1] > x1,
            record['centers'][:,0] < y2-1,
            record['centers'][:,1] < x2-1
        ])
        center_idxs = np.where(np.all(conditions,axis=0))[0]
    else:
        center_idxs = np.array([])

    if center_idxs.shape[0] > 0:
        centers = np.int32(record['centers'][center_idxs,:])
        centers[:,0] -= y1
        centers[:,1] -= x1
    else:
        centers = np.array([])

    return {
        'image':record['image'][x1:x2,y1:y2],
        'mask': record['mask'][x1:x2,y1:y2],
        'weight_map':record['weight_map'][x1:x2,y1:y2],
        'centers':centers}

def rotate_image(image,angle,interpolation=cv2.INTER_LINEAR):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image,rot_mat,
                            image.shape[1::-1],
                            flags=interpolation)
    return result

def record_rotation(record,angle):
    sh = record['image'].shape
    angle = angle % 360
    record = {
        'image':record['image'][:],
        'mask':record['mask'][:],
        'weight_map':record['weight_map'][:],
        'centers':record['centers'][:]}

    record['image'] = rotate_image(
        record['image'],angle)
    record['mask'] = rotate_image(
        record['mask'],angle,cv2.INTER_NEAREST)[:,:,np.newaxis]
    record['weight_map'] = rotate_image(
        record['weight_map'],angle)[:,:,np.newaxis]
    if record['centers'].shape[0] > 0:
        record['centers'][:,0],record['centers'][:,1] = rotate_coords(
            record['centers'][:,0]-sh[0]//2,
            record['centers'][:,1]-sh[1]//2,
            -angle
        )
        record['centers'][:,0] += sh[0]//2
        record['centers'][:,1] += sh[1]//2
        centers_out = []
        record['centers'] = np.array(centers_out)

    return record

class SegmentationDataset:
    """Carries segmentation datasets."""

    def __init__(self,
                 hdf5_file,
                 dimensions,
                 rel_keys=['image','mask'],
                 transform=None,
                 mode='full',
                 rotate_record=False):
        """

        """
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(self.hdf5_file, 'r')
        self.hf_keys = list(self.hf.keys())
        self.sizes = [self.hf[k]['image'].shape[:2] for k in self.hf_keys]
        self.idx_to_keys = {i:x for i,x in enumerate(self.hf_keys)}
        self.dimensions = dimensions
        self.size = (self.dimensions[2]-self.dimensions[0],
                     self.dimensions[3]-self.dimensions[1])

        self.rel_keys = rel_keys
        self.transform = transform
        self.mode = mode
        self.rotate_record = rotate_record

    def __len__(self):
        return len(self.hf)

    def keys(self):
        return list(self.hf.keys())

    def getitem(self,idx):
        def get_rotation_dimensions(he,wi,angle):
            theta = np.radians(angle)
            inter_h = np.maximum(
                int(np.ceil(np.sin(theta)*he+np.cos(theta)*wi)),out_h)
            inter_w = np.maximum(
                int(np.ceil(np.cos(theta)*he+np.sin(theta)*wi)),out_w)
            return inter_w,inter_h

        p = 1 # offset for two-step rotation

        if isinstance(idx,int):
            key = self.idx_to_keys[idx]
        else:
            key = idx
        record = self.hf[key]

        in_h,in_w,_ = record['image'].shape
        out_h,out_w = self.size

        if self.rotate_record == True:
            angle = np.random.randint(0,90)
            if angle == 0:
                out_x = np.random.randint(0,in_h - out_h)
                out_y = np.random.randint(0,in_w - out_w)
                record = select_region(
                    record,
                    [out_x,out_y,out_x+out_h,out_y+out_w])

            else:
                inter_h,inter_w = get_rotation_dimensions(out_h,out_w,angle)
                inter_x = np.random.randint(p,in_h - inter_h - p)
                inter_y = np.random.randint(p,in_w - inter_w - p)
                prerotation_dim = (
                    inter_x-p,inter_y-p,
                    inter_x + inter_h + p,inter_y + inter_w + p
                )
                postrotation_dim = (
                    ((inter_h+p) - out_h)//2,
                    ((inter_w+p) - out_w)//2,
                    out_h + ((inter_h+p) - out_h)//2,
                    out_w + ((inter_w+p) - out_w)//2
                )
                record = select_region(record,prerotation_dim)
                record = record_rotation(record,angle)
                record = select_region(record,postrotation_dim)

        else:
            out_x = np.random.randint(0,in_h - out_h)
            out_y = np.random.randint(0,in_w - out_w)
            dimensions = (out_x,out_y,out_x+out_h,out_y+out_w)

            record = select_region(record,dimensions)

        sample = {x:record[x] for x in self.rel_keys}
        sample['image_name'] = key

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getitem_segmentation(self,record):
        return {
            'image':record['image'],
            'mask':record['mask'],
            'weight_map':record['weight_map']}

    def __getitem__(self,idx):
        record = self.getitem(idx)
        if self.mode == 'full':
            record = record
        elif self.mode == 'segmentation':
            record = self.getitem_segmentation(record)
        if isinstance(idx,int):
            key = self.idx_to_keys[idx]
        else:
            key = idx
        record['image_name'] = key
        return record
