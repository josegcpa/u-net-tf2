source /hps/research/gerstung/josegcpa/projects/01IMAGE/tf2/bin/activate

mkdir -p test_results

for dataset in adden_1 adden_2 mll
do
    for DEPTH in 0.25 0.5 1.0
    do
        testing_set_file=../u-net/testing_set_files_$dataset
        python3 test-u-net.py\
            --dataset_path ../u-net/segmentation_dataset_all_cohorts.h5\
            --padding SAME\
            --depth_mult $DEPTH\
            --checkpoint_path checkpoints/u-net-$DEPTH/u-net-100 \
            --tta \
            --key_list $testing_set_file | grep TEST > test_results/tta_$dataset.100.$DEPTH
    done
done