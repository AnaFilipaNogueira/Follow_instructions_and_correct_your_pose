#!/usr/bin/bash
#​
#SBATCH --partition=gpu_min24GB       # Global GPU partition (i.e., any GPU)​gpu, v100_32GB, rtx2080ti_11GB
#SBATCH --job-name=fixmypose   # Job name​
#SBATCH -o slurm.%N.%j.out      # STDOUT​
#SBATCH -e slurm.%N.%j.err      # STDERR​
#SBATCH --qos=gpu_min24gb             #rtx2080ti, v100

python version.py
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

dataset=fixmypose
task=speaker

folders_speaker_eng_list=("normal") #options: "cutout_img0"  "cutout_imgtrg" "cutout_img0_imgtrg" #"normal" "normal_PTFalse"
nlpaug_choice_list=("delete_random")  #options: 'spelling_aug', 'delete_random', 'synonym_replace', 'sometimes', 'sequential'

for nlp_aug in "${nlpaug_choice_list[@]}"
do
    for name_folder_eng in "${folders_speaker_eng_list[@]}"
    do
        alteracoes=$name_folder_eng
        
        nlpaug_choice=$nlp_aug
        aug_p=0.5
        choose_stop_words=True 
        
        echo $alteracoes ----------------------------------------------------------------------------------------------------------

        cutout_p=0.5
        # Name of the model, used in snapshot
        name=${model}_2pixel_retrieval_${task}_${alteracoes}_${nlpaug_choice}_Stopwords_new

        if [ -z "$1" ]; then
            gpu=0
        else
            gpu=$1
        fi

        log_dir=$dataset/$task/$name
        mkdir -p snap_retrieval/$log_dir
        mkdir -p log_retrieval/$dataset/$task
        cp $0 snap_retrieval/$log_dir/run.bash
        cp -r src_retrieval snap_retrieval/$log_dir/src

        CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src_retrieval/main.py --output snap_retrieval/$log_dir \
            --maxInput 100  --worker 4 --train $task --dataset $dataset --alteracoes $alteracoes --cutout_p $cutout_p \
            --nlpaug_choice $nlpaug_choice --aug_p $aug_p --choose_stop_words $choose_stop_words \
            --batchSize 15 --hidDim 512 --dropout 0.5 \
            --seed 5555 \
            --optim adam --lr 1e-4 --epochs 50 \
            | tee log_retrieval/$log_dir.log
    done
done
