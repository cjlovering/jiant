#!/bin/sh
#$ -cwd
#$ -l short
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./user_config.sh;

# full sentence evaluation
# python main.py --config_file jiant/config/blimp/blimp_gpt2.conf --overrides="exp_name=blimp-gpt2,run_name=simplelm,target_tasks=blimp-simpleLM,input_module=gpt2-large"
python main.py --config_file jiant/config/blimp/blimp_bert.conf
