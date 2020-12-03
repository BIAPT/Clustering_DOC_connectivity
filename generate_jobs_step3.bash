#!/bin/bash

# We assume running this from the script directory
MODES=("dpli" "wpli")
FREQUENCIES=("alpha" "theta" "delta")
#HEALTHY=("Yes" "No")
#STEPS=("01" "10")
#FREQUENCIES=("alpha")
HEALTHY=("Yes")
STEPS=("10")
REP=2

for mode in ${MODES[@]}; do
    for frequency in ${FREQUENCIES[@]}; do
        for healthy in ${HEALTHY[@]}; do 
			for steps in ${STEPS[@]}; do 
				for ((r=1;r<=REP;r++)); do
					analysis_param="${mode}_${frequency}_${healthy}_${steps}_${r}"
					echo "${analysis_param}"
					sbatch --export=ANALYSIS_PARAM=$analysis_param $1
				done
			done
		done
    done
done
