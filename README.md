# U-Net implementation in TF2.0

The U-Net model needs barely any introduction, but it was first introduced by Ronneberger in 2014/2015 ([paper here](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)) and then later repackaged in 2018 for a much deserved Nature methods [paper](https://www.nature.com/articles/s41592-018-0261-2).

The core concept of the U-Net is fairly simple - a fully convolutional network which combines a U-shaped/encoder-decoder/hourglass-shaped architecture (features are retrieved sequentially until a rich representation is achieved, followed by a reconstruction of the initial resolution) with skip-connections (earlier layers in the encoder are concatenated to later layers in the decoder and convolved) for segmentation. It has recurringly been the most popular option for a number of segmentation tasks in biological problems. This implementation is fairly standard and features few additions (squeeze and excite layers being one of them).

The code for training in this repository relies on the creation of a SegmentationDataset stored in hdf5 format which reduces latency in the I/O, I will make scripts for this available at a later time. In the creation of this dataset, weight maps, as well as bounding boxes and polygons, are calculated for all the image-mask pairs.
