#!/bin/bash
#SBATCH --job-name SAVI_all
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=48GB
#SBATCH -p tenenbaum
#SBATCH --mem=48G
#SBATCH --out=savi_tex=all_shapes=2,3_slots=4.out

run_name="tex=noise_shapes=1,2,3_slots=4"
python train_savi.py --data_dir /om2/user/yyf/CommonFate/scenes/ \
        --top_level noise \
        --sub_level superquadric_1 superquadric_2 superquadric_3 \
        --log_dir /om2/user/yyf/GestaltVision/runs/SAVI/${run_name} \
        --checkpoint_dir /om2/user/yyf/GestaltVision/saved_models/SAVI/${run_name} \
        --batch_size 16 \
        --load_latest_model

