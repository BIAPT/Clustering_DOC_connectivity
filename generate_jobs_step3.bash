#!/bin/bash

# We assume running this from the script directory
#MODES=("dpli" "wpli" "aec")
MODES=("dpli" "wpli")
FREQUENCIES=("alpha" "theta" "delta")
STEPS=("01" "10")
#STEPS=("01")
#FREQUENCIES=("alpha")
REP=5

for mode in ${MODES[@]}; do
    for frequency in ${FREQUENCIES[@]}; do
		for steps in ${STEPS[@]}; do 
			for ((r=1;r<=REP;r++)); do
				analysis_param="${mode}_${frequency}_${steps}_${r}"
				echo "${analysis_param}"
				sbatch --export=ANALYSIS_PARAM=$analysis_param $1
			done
		done
    done
done
