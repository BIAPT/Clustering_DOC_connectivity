#!/bin/bash

# We assume running this from the script directory
#MODES=("dpli" "wpli")
#FREQUENCIES=("alpha" "theta" "delta")
#HEALTHY=("Yes" "No")
#STEPS=("01" "10")
MODES=("dpli")
FREQUENCIES=("alpha")
HEALTHY=("Yes")
STEPS=("01")

for mode in ${MODES[@]}; do
    for frequency in ${FREQUENCIES[@]}; do
        for healthy in ${HEALTHY[@]}; do 
			for steps in ${STEPS[@]}; do 
				analysis_param="${mode}_${frequency}_${healthy}_${steps}"
				echo "${analysis_param}"
				sbatch --export=ANALYSIS_PARAM=$analysis_param $1
			done
		done
    done
done