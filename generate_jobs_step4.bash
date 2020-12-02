#!/bin/bash

# We assume running this from the script directory
#MODES=("dpli" "wpli")
#FREQUENCIES=("alpha" "theta" "delta")
#HEALTHY=("Yes" "No")
#STEPS=("01" "10")
#VALUE=("Diag" "Prog")
MODES=("dpli")
FREQUENCIES=("alpha")
HEALTHY=("Yes")
STEPS=("10")
VALUE=("Prog")

for mode in ${MODES[@]}; do
    for frequency in ${FREQUENCIES[@]}; do
        for healthy in ${HEALTHY[@]}; do 
			for steps in ${STEPS[@]}; do 
				for value in ${VALUE[@]}; do
					analysis_param="${mode}_${frequency}_${healthy}_${steps}_${value}"
					echo "${analysis_param}"
					sbatch --export=ANALYSIS_PARAM=$analysis_param $1
				done
			done
		done
    done
done
