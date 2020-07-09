#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/
#$ -t 1-1
#$ -tc 4

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./user_config.sh;

# full sentence evaluation
if [ "$SGE_TASK_ID" -eq 1 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement,run_name=1,target_tasks=distractor_agreement_relational_noun_probing-0.5-good,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 2 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement,run_name=2,target_tasks=distractor_agreement_relational_noun_probing-0.5-good-blimp,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 3 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement,run_name=3,target_tasks=distractor_agreement_relational_noun_probing-1.0-good,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 4 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement,run_name=4,target_tasks=distractor_agreement_relational_noun_probing-1.0-good-blimp,transfer_paradigm=finetune"
fi