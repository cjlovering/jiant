#!/bin/sh
#$ -cwd
#$ -l short
#$ -l gpus=1
#$ -e ./logs/
#$ -o ./logs/

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./user_config.sh; python main.py --config_file jiant/config/snli_eval_nep.conf
