python3 test-u-net.py\
    --dataset_path ../u-net/segmentation_dataset.h5\
    --padding SAME\
    --depth_mult 0.25\
    --checkpoint_path checkpoints/u-net-0.25/u-net-100\
    --key_list ../u-net/training_set_files
