for test_file in test_results/*
do
    tta=$(echo $test_file | grep tta | wc -l)
    refine=$(echo $test_file | grep refine | wc -l)
    dataset=$(basename $test_file | cut -d '.' -f 1)
    epoch=$(basename $test_file | cut -d '.' -f 2)
    depth=$(basename $test_file | cut -d '.' -f 3-10)
    if [[ $tta == 1 ]]
    then
        dataset=$(echo $dataset | sed 's/tta_//')
    fi
    if [[ $refine == 1 ]]
    then
        dataset=$(echo $dataset | sed 's/ttarefine_//')
    fi
    cat $test_file | grep TEST | awk -v id_str="$dataset,$epoch,$depth,$tta,$refine" '{print id_str "," $0}'
done
