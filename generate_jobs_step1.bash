#!/bin/bash

# We assume running this from the script directory
IDS=("MDFA03" "MDFA05" "MDFA06" "MDFA07" "MDFA10" "MDFA11" "MDFA12" "MDFA15" "MDFA17" "WSAS02" "WSAS05" "WSAS07" "WSAS09" "WSAS10" "WSAS11" "WSAS12" "WSAS13" "WSAS15" "WSAS16" "WSAS17" "WSAS18" "WSAS19" "WSAS20" "WSAS22" "WSAS23" "AOMW03" "AOMW04" "AOMW08" "AOMW22" "AOMW28" "AOMW31" "AOMW34" "AOMW36")
FREQUENCY=("alpha" "theta" "delta")
STEPS=("01" "10")

for id in ${IDS[@]}; do
    for frequency in ${FREQUENCY[@]}; do
        for steps in ${STEPS[@]}; do 
            analysis_param="${id}_${frequency}_${steps}"
            echo "${analysis_param}"
            sbatch --export=ANALYSIS_PARAM=$analysis_param $1
        done

    done

done
