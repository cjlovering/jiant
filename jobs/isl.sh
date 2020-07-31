#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/
#$ -t 1-1
#$ -tc 5

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./user_config.sh;

# probing
if [ "$SGE_TASK_ID" -eq 1 ]
then
python3 main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=isl,run_name=strong_probing,target_tasks=isl_probing_strong,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 5 ]
then
python3 main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=isl,run_name=weak_probing,target_tasks=isl_probing_weak,transfer_paradigm=finetune"
fi

# fine-tuning
if [ "$SGE_TASK_ID" -eq 2 ]
then
python3 main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=isl,run_name=0,target_tasks=isl_finetune_0,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 3 ]
then
python3 main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=isl,run_name=1,target_tasks=isl_finetune_1,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 4 ]
then
python3 main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=isl,run_name=5,target_tasks=isl_finetune_5,transfer_paradigm=finetune"
fi