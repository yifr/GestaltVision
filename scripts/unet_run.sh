#!/bin/bash
#SBATCH --job-name UNET_all
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=48GB
#SBATCH -p tenenbaum
#SBATCH --mem=48G
#SBATCH --out=unet_trgt=normals_tex=all_shapes=2,3.out

run_name="unet_trgt=normals_tex=all_shapes=2,3"
python train_unet.py --data_dir /om2/user/yyf/CommonFate/scenes/ \
        --target normals \
        --top_level voronoi noise \
        --sub_level superquadric_2 superquadric_3 \
        --log_dir /om2/user/yyf/GestaltVision/runs/UNet/${run_name} \
        --checkpoint_dir /om2/user/yyf/GestaltVision/saved_models/UNet/${run_name} \
        --frames_per_scene 6 \
        --n_classes 3 \
        --resize 256 \
        --batch_size 16 \

