#!/bin/bash
#SBATCH --job-name gestalt_vision
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 10:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p tenenbaum
#SBATCH --mem=24G

python run.py --config configs/vqvae.json