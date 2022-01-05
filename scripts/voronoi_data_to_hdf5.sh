#!/bin/bash
#SBATCH --job-name voronoi_transfer
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -p normal
#SBATCH --mem=24G
#SBATCH --array=1-4

python data/preprocessing/data_to_hdf5.py --top_level voronoi --sub_level superquadric_${SLURM_ARRAY_TASK_ID} --output_dir /om2/user/yyf/CommonFate/scenes --data_dir /om/user/yyf/CommonFate/scenes --image_passes images masks flows depths
