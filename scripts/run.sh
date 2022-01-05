#!/bin/bash
#SBATCH --job-name SAVI
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 12:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=48GB
#SBATCH -p tenenbaum
#SBATCH --mem=24G

python train_savi.py
