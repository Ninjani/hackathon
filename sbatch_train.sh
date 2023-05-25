# Run as ./sbatch_train.sh <job_name> <config_file>

# Retrieve command name from the command line
JOB_NAME=$1
CONFIG_FILE=$2

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=hackathon_"$JOB_NAME"
#SBATCH --nodes=1             
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=rtx8000,pascal
#SBATCH --output="$JOB_NAME".out
#SBATCH --qos=1day

# activate conda env
export PATH=$HOME/miniconda3/bin:$PATH
source activate hackathon

srun python main.py fit -c $CONFIG_FILE
EOF