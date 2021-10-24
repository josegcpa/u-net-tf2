for test_file in test_results/*
do 
    tta=$(echo $test_file | grep tta | wc -l)
    dataset=$(basename $test_file | cut -d '.' -f 1)
    depth=$(basename $test_file | cut -d '.' -f 2-10)
    if [[ $tta == 1 ]]
    then
        dataset=$(echo $dataset | sed 's/tta_//')
    fi
    cat $test_file | grep TEST | awk -v id_str="$dataset,$depth,$tta" '{print $0 "," id_str}'
done