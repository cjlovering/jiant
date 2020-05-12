#!/bin/sh
#$ -cwd
#$ -l short
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/
#$ -t 1-13
#$ -tc 3

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./user_config.sh;

# full sentence evaluation
if [$SGE_TASK_ID -eq 1]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.5-good,target_tasks=distractor_agreement_relational_noun_probing-0.5-good"
fi

if [$SGE_TASK_ID -eq 2]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.5-bad,target_tasks=distractor_agreement_relational_noun_probing-0.5-bad"
fi

if [$SGE_TASK_ID -eq 3]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.6-good,target_tasks=distractor_agreement_relational_noun_probing-0.6-good"
fi

if [$SGE_TASK_ID -eq 4]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.6-bad,target_tasks=distractor_agreement_relational_noun_probing-0.6-bad"
fi

if [$SGE_TASK_ID -eq 5]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.7-good,target_tasks=distractor_agreement_relational_noun_probing-0.7-good"
fi

if [$SGE_TASK_ID -eq 6]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.7-bad,target_tasks=distractor_agreement_relational_noun_probing-0.7-bad"
fi

if [$SGE_TASK_ID -eq 7]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.8-good,target_tasks=distractor_agreement_relational_noun_probing-0.8-good"
fi

if [$SGE_TASK_ID -eq 8]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.8-bad,target_tasks=distractor_agreement_relational_noun_probing-0.8-bad"
fi

if [$SGE_TASK_ID -eq 9]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.9-good,target_tasks=distractor_agreement_relational_noun_probing-0.9-good"
fi

if [$SGE_TASK_ID -eq 10]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-0.9-bad,target_tasks=distractor_agreement_relational_noun_probing-0.9-bad"
fi

if [$SGE_TASK_ID -eq 11]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-1.0-good,target_tasks=distractor_agreement_relational_noun_probing-1.0-good"
fi

if [$SGE_TASK_ID -eq 12]
then
python main.py --config_file jiant/config/blimp/blimp_bert.conf --overrides="exp_name=blimp-bert-probing,run_name=distractor_agreement_relational_noun_probing-1.0-bad,target_tasks=distractor_agreement_relational_noun_probing-1.0-bad"
fi