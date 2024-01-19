#!/usr/bin/bash
#​
#SBATCH --partition=gpu_min8GB          # Global GPU partition (i.e., any GPU)​gpu, v100_32GB, rtx2080ti_11GB
#SBATCH --job-name=fixmypose_hindi      # Job name​
#SBATCH -t 10-10:00                     # Time (D-HH:MM)​
#SBATCH -o slurm.%N.%j.out              # STDOUT​
#SBATCH -e slurm.%N.%j.err              # STDERR​
#SBATCH --qos=gpu_min8gb                #rtx2080ti, v100

python version.py

export LC_ALL=C.UTF-8
#export LANG=C.UTF-8

dataset=fixmypose

# Main metric to use
metric=CIDEr

task=speaker

folders_speaker_eng_list=("cutout_img0" "cutout_img1" "cutout_img0_img1") #"cutout_img0" "cutout_img0_img1" "cutout_img1") #"normal" "normal_PTFalse"

for name_folder_eng in "${folders_speaker_eng_list[@]}"
do
    alteracoes=$name_folder_eng
    
    echo $alteracoes ----------------------------------------------------------------------------------------------------------

    # Name of the model, used in snapshot
    name=${model}_2pixel_${task}_${alteracoes}_hindi0.5_NEW_PTFalse

    if [ -z "$1" ]; then
        gpu=0
    else
        gpu=$1
    fi

    log_dir=$dataset/$task/$name
    mkdir -p snap_hindi/$log_dir
    mkdir -p log_hindi/$dataset/$task
    cp $0 snap_hindi/$log_dir/run.bash
    cp -r src_hindi snap_hindi/$log_dir/src

    CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src_hindi/main.py --output snap_hindi/$log_dir \
        --maxInput 40 --metric $metric  --worker 4 --train $task --dataset $dataset --alteracoes $alteracoes --cutout_p 0.5\
        --batchSize 45 --hidDim 512 --dropout 0.5 \
        --seed 9595 \
        --optim adam --lr 1e-4 --epochs 500 \
        | tee log_hindi/$log_dir.log
done