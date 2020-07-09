#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/
#$ -t 1-8
#$ -tc 8

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./user_config.sh;

# full sentence evaluation
if [ "$SGE_TASK_ID" -eq 1 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=1,target_tasks=distractor_agreement_relational_noun_probing-0.5-good,transfer_paradigm=frozen"
fi

if [ "$SGE_TASK_ID" -eq 2 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=2,target_tasks=distractor_agreement_relational_noun_probing-0.5-bad,transfer_paradigm=frozen"
fi

if [ "$SGE_TASK_ID" -eq 3 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=3,target_tasks=distractor_agreement_relational_noun_probing-1.0-good,transfer_paradigm=frozen"
fi

if [ "$SGE_TASK_ID" -eq 4 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=4,target_tasks=distractor_agreement_relational_noun_probing-1.0-bad,transfer_paradigm=frozen"
fi

if [ "$SGE_TASK_ID" -eq 5 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=5,target_tasks=distractor_agreement_relational_noun_probing-0.5-good,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 6 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=6,target_tasks=distractor_agreement_relational_noun_probing-0.5-bad,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 7 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=7,target_tasks=distractor_agreement_relational_noun_probing-1.0-good,transfer_paradigm=finetune"
fi

if [ "$SGE_TASK_ID" -eq 8 ]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=agreement_probing,run_name=8,target_tasks=distractor_agreement_relational_noun_probing-1.0-bad,transfer_paradigm=finetune"