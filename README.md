# Altered dynamic FC in DOC
about the project: 
### Altered dynamic properties of functional connectivity in patients in a Disorder of Consciousness 
This project aims to identify diagnostic and prognostic value of time-resolved functional connectivity. 

To reproduce the analysis or use the code with other data, follow the steps below

## Preparation 
- The code is optimized to run on a remote computation resource, in our case Compute Canada. Especially the calculation of the time-resolved functional connectivity and the calculation of the stability index are very time-consuming on a private computer. The code in its recent shape is optimized for Compute Canada to run on the Beluga Cluster, but can be run on a private computer too. The current documentation steps are for the remote resource. 

- Create a project folder inside your account. Inside there, create folders named "data" and "results"

- clone this repo inside your personal directory on Beluga. 

- clone this code, as well as Neuroalgo toolbox in your Compute Canada account (also in the BIAPT GitHub)

- upload all data as .mat files inside your folder "data" using [Globus](https://globus.computecanada.ca/) or scp

- Make sure you have Matlab 2020a with the parallel toolbox installed on your machine. 

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



## Step 2: Generate the Feature DataFrame
- create a features folder in your results folder on Compute Canada
- Navigate into the folder `STEP2_load-and-store-fc-data`. Once there open the `extract_features.sl` and modify the parameters to match the resource you want to use and your account on compute Canada. 
- Once ready you can run the following commands: `sbatch extract_features.sl` assuming the cluster variable is already assigned as show in step 1.
- this will output your time resolved functional connectivity matrices for all your conditions in the features folder (as pickle and csv)
- download these files and put them into your local data folder



## Step 3: Find the number of clusters

- Open the generate_jobs_step3.bash and adapt the parameters and wanted number of repetitions

  (start maybe with a smaller number of repetitions to make sure everything works properly)

- adapt the job_staility.sl in the step3 folder. 

- adapt the output directory in the compute_stability.py (a lot of .txt files will be saved there)

- Run the generate_jobs_step3.bash by typing: ` bash generate_jobs_step3.bash step_3_find_number_of_clusters/job_staility.sl` This will submit one job per condition and repetition. It will take a bit of time. 

- Once this is done, you can download the folder with all .txt files, move it to the data folder of your local GitHub folder copy and run "summarize_SI.py " This code will output a pdf with the figures

**Word of Caution:** The chosen way is not the most optimal method. I have tried to parallelize it in a more optimal way but it did not work (there were some problems to fit the model in the parallelized job) If few conditions or repetitions are needed, this code can be easily adapted to run on a PC.  



## Step 4: perform K-means clustering

- this step is executable only on a private machine and is not optimized for Compute Canada. 
- open the PCA_DOC_Cluster_Combined.py and adapt the output input files 
- open the helper_function general information and adapt the containing information and datapath (these functions are some kind of data-loader)
- run the PCA_DOC_Cluster_Combined.py 
- this outputs a pdf with all images, statistics and results. 

 

#### Structure of the analysis 

- the PCA_DOC_Cluster_Combined.py script access the helper function  general information to load the data
- the PCA_DOC_Cluster_Combined.py script access the helper function  "process properties" to compute all dynamic properties 
- the PCA_DOC_Cluster_Combined.py script access the helper function  "statistics " to perform the statistics
- the PCA_DOC_Cluster_Combined.py script access the helper function  "visualize" to plot all results