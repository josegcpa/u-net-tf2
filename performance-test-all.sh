mkdir -p performance

for prediction_key in prediction prediction_tta prediction_tta_refine
do
    for prediction in predictions/*
    do
        root=$(echo $prediction | cut -d / -f 2 | cut -d . -f 1)
        root_out=$(echo $prediction | cut -d / -f 2 | cut -d . -f 1,2,3)
        if [[ $root == "adden_1" ]]
        then
            truth_path=../u-net/segmentation_dataset.h5
        elif [[ $root == "adden_2" ]]
        then
            truth_path=../u-net/segmentation_dataset_adden_2.h5
        elif [[ $root == "mll" ]]
        then
            truth_path=../u-net/segmentation_dataset_munich.h5
        fi
        bsub -n 4 -M 4000 -o /dev/null -e /dev/null \
            "python3 performance-test.py \
                --prediction_path $prediction \
                --truth_path $truth_path \
                --n_iou_threshold 200 \
                --prediction_key $prediction_key > performance/$root_out.$prediction_key.csv"
    done
done
