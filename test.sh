mkdir -p test_results

for DEPTH in 0.25 0.5 1.0
do
    for dataset in adden_1 adden_2 mll
    do
        testing_set_file=../u-net/testing_set_files_$dataset
        python3 test-u-net.py\
            --dataset_path ../u-net/segmentation_dataset_all_cohorts.h5\
            --padding SAME\
            --depth_mult $DEPTH\
            --checkpoint_path checkpoints/u-net-$DEPTH/u-net-100 \
            --key_list $testing_set_file | grep TEST > test_results/$dataset.$DEPTH

        testing_set_file=../u-net/testing_set_files_$dataset
        python3 test-u-net.py\
            --dataset_path ../u-net/segmentation_dataset_all_cohorts.h5\
            --padding SAME\
            --depth_mult $DEPTH\
            --checkpoint_path checkpoints/u-net-$DEPTH/u-net-100 \
            --tta \
            --key_list $testing_set_file | grep TEST > test_results/tta_$dataset.$DEPTH
    done
done
