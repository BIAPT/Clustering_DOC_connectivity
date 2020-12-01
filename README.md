# Altered dynamic FC in DOC
about the project: 
### Altered dynamic properties of functional connectivity in patients in a Disorder of Consciousness 
This project aims to identify diagnostic and prognostic value of time-resolved functional connectivity. 

To reproduce the analysis or use the code with other data, follow the steps below

## Preparation 
- The code is optimized to run on a remote computation resource, in our case Compute Canada. Especially the calculation of the time-resolved functional connectivity and the calculation of the stability index are very time-consuming on a private computer. The code in its recent shape is optimized for Compute Canada to run on the Beluga Cluster, but can be run on a private computer too. The current documentation steps are for the remote resource. 

- Create a project folder inside your account. Inside there, create folders named "data" and "results"

- clone this repo inside your personal directory on Beluga. 

- clone this code, as well as Neuroalgo toolbox in your Compute Canada account (also in the BIAPT github)

- upload all data as .mat files inside your folder "data" using [Globus](https://globus.computecanada.ca/) or scp

- Make sure you have Matlab 2020a with the parallel toolbox intalled on your machine. 

  ## Notes:

  - BEFORE YOU RUN A JOB: 

    Open the .sl file which will run your job and change your e-mail address (slurm file which is there to submit your job. Look at the documentation of Compute Canada if this is new for you) 

    This will send you a notification when your job begins and ends. 

  - All jobs can be executed from the main folder by typing  `sbatch jobname.sl `  the job will be submitted to the cluster. Please make sure to adapt all .sl files to contain your e-mail address and supervisors account name. 

  - Once the job is running it will output a slurmXXXXXXX.out file. This file contains the job output. When a job fails have a look into this jobID.out file to identify the error.   

  - almost all files were set up to be run on compute Canada. All steps can be run on a private computer in theory without complication, but will take a considerable amount of time (especially step0). 

## Step 1: Calculate time-resolved functional Connectivity

This step will calculate the time-resolved functional connectivity with a step size of 10 and 1 and a window size of 10. The functional connectivity measures are wPLI and dPLI. This is the only part of the analysis run in matlab. You have two ways of running the analysis:  

#### way1:  

- navigate to the results folder and create 2 subdirectories called "wPLI" and "dPLI" inside each of them create the subfolders "step10" and "step01"  (or others depending on your analysis parameters)
-  navigate to`step_0_generate_graphs/`.
- Move to`STEP1_calculate-time-resolved-functional-connectivity ` Choose wPLI, open the `dPLI_claculation.sl` and modify the parameters to match the resource you want to use and your account on Compute Canada.
- Open the `dPLI_for_time_resolved.m` file and modify the parameter relating to your cluster setup and path. There are some path that needs to be modified in order for the input/output to make sense in your section of the cluster. 
- Once ready you can run the following commands: `sbatch  dPLI_claculation.sl`, this will send the job to your cluster
- Repeat the same procedure for `generate_wpli_graphs`. You do not have to wait for the dPLI graph to be done before you start running the wPLI computation.

#### way 2: (more time to set up but much faster to run) 

- follow the instructions on "Using the MATLAB Parallel Server" https://docs.computecanada.ca/wiki/MATLAB to set up Metlab 2020a (other versions are not supported) 
- open  `dPLI_for_time_resolved.m` on your private Matlab environment but adapt the paths for the remote directory.  
- adapt also the `ccSBATCH.m `  to correspond to your file directory
- run the commands `cluster = parcluster('beluga')` in your local python environment. After this, run `ccSBATCH.submitTo(cluster)`
- If everything is set up right, your local Matlab environment will ask you for the password and the jobs will be submitted to the cluster and run on several nodes in parallel. 
- repeat the same steps for wPLI

**Word of Caution:** The speed up that you can gain from the parallelization really depends on what is the availability of the cluster. If you try to use up to 960 cores you might wait a long time before it becomes available (I've waited 8+h and still wasn't scheduled). However, if you use only 40 cores on 1 node you will most likely get scheduled right away. There is a balance to strike and it is still not obvious what is the best course of action. Sometime you get lucky sometime you don't.



## Step 2: Generate the Feature Dataframe
- create a features folder in your results folder on Compute Canada
- Navigate into the folder `STEP2_load-and-store-fc-data`. Once there open the `extract_features.sl` and modify the parameters to match the resource you want to use and your account on compute Canada. 
- Once ready you can run the following commands: `sbatch extract_features.sl` assuming the cluster variable is already assigned as show in step 1.
- this will output your time resolved functional connectivity matrices for all your conditions in the features folder (as pickle and csv)



## TODO 

## Step 2: Run the model selection with LOSO cross validation

- Open the commons.py file and make sure that the input and output are correct for your HPC setup and define which epoch, graph and feature category you want to select

- crate a folder called  "models" in the "results" folder

- Then adapt and run `step_2a_run_all_models.sl` 

  --> this should fill the "models" folder with a summary of all possible models

- after the job has finished, adapt and run `step_2b_visualize_all_models.sl`

  --> this should output a .pdf and .txt with the summary of all models. The pdf contains the visual summary of each model with the averaged accuracy and f1 score in the title. The .txt outputs all accuracies and the model with the highest accuracy and F1. 

  --> Download the pdf and txt and have a look at them

- open the commons.py and select the best_model to be the model with the highest accuracy. 

## Step 3: Run the Final model and visualize the summary

- Open the commons.py file and make sure that the parameter "best_model" is the model you have selected as your best performing model. (based on the previous step) 

- crate a folder called  "final_models" in the "results" folder

- Then adapt and run `step_3a_run_final_model.sl` 

  --> this should fill the "final_models" folder with a summary of the final model's performance in all conditions

- after the job has finished, adapt and run `step_3b_visualize_models.sl`

  --> this should output several .csv files with the accuracy and F1 score for the model in all conditions  

  --> Download these csv files. These are the FINAL ACCURACIES

## Step 4: Run the Final model bootstrap and permutation

- Open the commons.py file and make sure that the parameter "best_model" is the model you have selected as your best performing model. (based on the previous step) 

- crate a folder called  "bootstrap" and "permutation" in the "results" folder

- Inside the step 4 folder, adapt  `job_bootstarp.sl and job_permutation.sl`  to contain your e-mail address

- navigate back to the subfolder  and open "generate_jobs.bash". This bash code contains all conditions you'll run the analysis on. It works like a for loop within a job but submits one individual job for each iteration. This has the advantage of having lots of small jobs instead of one giant job. This reduces the waiting time significantly.  

- Adapt the conditions you want to run and run the following command ` bash generate_jobs.bash step_4_characterize_classification/job_permutation.sl` 

  --> this should fill the "permutation" folder with excel files for all conditions 

- Adapt the conditions you want to run and run the following command ` bash generate_jobs.bash step_4_characterize_classification/job_bootstrap.sl` 

  --> this should fill the "bootstrap" folder with excel files for all conditions 

- You can now download these folders

## Step 5: Visualize the features

- Open the commons.py file and make sure that the parameter "best_model" is the model you have selected as your best performing model. (based on the previous step) 

- adapt and submit the job ` job_step_5_extract_weights.sl` 

  --> it will output you two .csv files  called "feature_weights" + stepsize

- download these and run the figure generation on your own computer to monitor the process: follow the next steps to do so

- copy the feature_weight csv into the step_5_generate_figures/plotting folder. 

- create an empty folder called "figures" (this is where all figures will be saved") 

- Open and adapt ` plot_brain_weights.m`  

  --> this will output many .fig files in the "figures" folder 

  (the parameter features corresponds to the feature which is selected when both graphs are inside the classifier)

  
