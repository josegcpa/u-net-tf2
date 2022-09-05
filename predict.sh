mkdir -p predictions

for EPOCH in 100
do
    for dataset in adden_1 adden_2 mll
    do
        for DEPTH in 0.25 0.5 1.0
        do
            testing_set_file=../u-net/testing_set_files_$dataset
            python3 predict-u-net.py\
                --input_path ../u-net/segmentation_dataset_all_cohorts.h5\
                --output_path predictions/$dataset.$DEPTH.h5\
                --padding SAME\
                --depth_mult $DEPTH\
                --checkpoint_path checkpoints/u-net-$DEPTH/u-net-$EPOCH \
                --key_list $testing_set_file | grep TEST > test_results/$dataset.$EPOCH.$DEPTH
        done
    done
done
