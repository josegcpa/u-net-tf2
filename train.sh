DEPTH=$1

python3 train-u-net.py \
    --dataset_path ../u-net/segmentation_dataset.h5 \
    --padding SAME \
    --input_height 512 \
    --input_width 512 \
    --number_of_epochs 50 \
    --beta_l2_regularization 0.005 \
    --learning_rate 0.01 \
    --depth_mult $DEPTH \
    --save_checkpoint_steps 500 \
    --save_checkpoint_folder checkpoints/u-net-$DEPTH \
    --save_summary_folder summaries/u-net-$DEPTH \
    --learning_rate 0.01 \
    --brightness_max_delta 0.125 \
    --saturation_lower 0.7 \
    --saturation_upper 1.3 \
    --hue_max_delta 0.1 \
    --contrast_lower 0.7 \
    --contrast_upper 1.3 \
    --noise_stddev 0.005 \
    --blur_probability 0.001 \
    --blur_size 1 \
    --blur_mean 0 \
    --blur_std 0.005 \
    --elastic_transform_p 0.3 \
    --discrete_rotation \
    --min_jpeg_quality 1 \
    --max_jpeg_quality 1 \
    --key_list ../u-net/training_set_files \
    --validation_iterations 15 \
    --log_every_n_steps 250 \
    --batch_size 2 \
    --truth_only
