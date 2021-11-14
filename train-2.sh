source /hps/research/gerstung/josegcpa/projects/01IMAGE/tf2/bin/activate

DEPTH=$1

python3 train-u-net.py \
    --dataset_path ../u-net/segmentation_dataset.h5 \
    --padding SAME \
    --input_height 512 \
    --input_width 512 \
    --number_of_epochs 100 \
    --beta_l2_regularization 0.005 \
    --learning_rate 0.0001 \
    --depth_mult $DEPTH \
    --save_checkpoint_steps 500 \
    --save_checkpoint_folder checkpoints/u-net-$DEPTH/u-net \
    --save_summary_folder summaries/u-net-$DEPTH \
    --brightness_max_delta 0.1 \
    --saturation_lower 0.9 \
    --saturation_upper 1.1 \
    --hue_max_delta 0.05 \
    --contrast_lower 0.9 \
    --contrast_upper 1.1 \
    --noise_stddev 0.005 \
    --blur_probability 0.001 \
    --blur_size 1 \
    --blur_mean 0 \
    --blur_std 0.005 \
    --elastic_transform_p 0.3 \
    --discrete_rotation \
    --min_jpeg_quality 90 \
    --max_jpeg_quality 100 \
    --key_list ../u-net/training_set_files \
    --validation_iterations 20 \
    --log_every_n_steps 250 \
    --batch_size 4 \
    --truth_only
