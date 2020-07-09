#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/
#$ -t 1-4
#$ -tc 4

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./user_config.sh;

# full sentence evaluation
if [ "$SGE_TASK_ID" -eq 1 ]
then
python main.py --config_file jiant/config/blimp/blimp_lstm_finetune.conf --overrides="exp_name=blimp-lstm-frozen,run_name=distractor_agreement_relational_noun_probing-0.5-good,target_tasks=distractor_agreement_relational_noun_probing-0.5-good"
fi

if [ "$SGE_TASK_ID" -eq 2 ]
then
python main.py --config_file jiant/config/blimp/blimp_lstm_finetune.conf --overrides="exp_name=blimp-lstm-frozen,run_name=distractor_agreement_relational_noun_probing-0.5-bad,target_tasks=distractor_agreement_relational_noun_probing-0.5-bad"
fi

if [ "$SGE_TASK_ID" -eq 3 ]
then
python main.py --config_file jiant/config/blimp/blimp_lstm_finetune.conf --overrides="exp_name=blimp-lstm-frozen,run_name=distractor_agreement_relational_noun_probing-1.0-good,target_tasks=distractor_agreement_relational_noun_probing-1.0-good"
fi

if [ "$SGE_TASK_ID" -eq 4 ]
then
python main.py --config_file jiant/config/blimp/blimp_lstm_finetune.conf --overrides="exp_name=blimp-lstm-frozen,run_name=distractor_agreement_relational_noun_probing-1.0-bad,target_tasks=distractor_agreement_relational_noun_probing-1.0-bad"
fi