python3 train-u-net.py \
    --dataset_path ../u-net/segmentation_dataset.h5 \
    --padding SAME \
    --input_height 512 \
    --input_width 512 \
    --number_of_epochs 50 \
    --beta_l2_regularization 0.005 \
    --learning_rate 0.0005 \
    --depth_mult 0.25 \
    --save_checkpoint_steps 500 \
    --save_checkpoint_folder checkpoints/u-net-0.25 \
    --saturation_lower 0.9 \
    --saturation_upper 1.1 \
    --hue_max_delta 0.1 \
    --contrast_lower 0.9 \
    --contrast_upper 1.1 \
    --key_list ../u-net/training_set_files \
    --validation_iterations 15 \
    --log_every_n_steps 250 \
    --batch_size 4 \
    --truth_only 