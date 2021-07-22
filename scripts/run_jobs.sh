#! /bin/bash

conda activate chemprop
clock=$(date '+%Y%m%d-%H%M%S')

###############################################################################
########################## Set Default Job Arguments ##########################
###############################################################################
N_TRIALS=10               # -k
N_ENSEMBLES=5             # -n
MAX_RUNS_PER_GPU=3        # -g
EPOCHS=100                # -e
BATCH_SIZE=50             # -b
DATA_TYPE="regression"    # -t
METRIC="rmse"             # -s
REG_COEFF=1.0             # -v
ATOMISTIC=0               # -a

ACTIVE_LEARNING=1         # -c
AL_LOOPS=10               # -l
AL_STRATEGY=( "explorative_greedy" "random" "score_greedy" )  # -y
AL_INIT_RATIO=0.1         # -r
LOG_PREFIX="/data/logs"   # -o
RESTART_INTERVAL=30       # -i

METHODS=( "evidence" "dropout" "ensemble" )  # -m
DATASETS=( "qm9" )        # -d
EXTRA_ARGS=""             # -x
###############################################################################


###############################################################################
############################ Read Command Arguments ###########################
###############################################################################
while getopts k:n:g:e:b:t:s:v:a:c:l:y:r:o:i:m:d:x: flag
do
    case "${flag}" in
        k) N_TRIALS=${OPTARG};;
        n) N_ENSEMBLES=${OPTARG};;
        g) MAX_RUNS_PER_GPU=${OPTARG};;
        e) EPOCHS=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        t) DATA_TYPE=${OPTARG};;
        s) METRIC=${OPTARG};;
        v) REG_COEFF=${OPTARG};;
        a) ATOMISTIC=${OPTARG};;
        c) ACTIVE_LEARNING=${OPTARG};;
        l) AL_LOOPS=${OPTARG};;
        y) AL_STRATEGY=(${OPTARG});;
        r) AL_INIT_RATIO=${OPTARG};;
        o) LOG_PREFIX=${OPTARG};;
        i) RESTART_INTERVAL=${OPTARG};;
        m) METHODS=(${OPTARG});;
        d) DATASETS=(${OPTARG});;
        x) EXTRA_ARGS=${OPTARG};;
    esac
done
LOG_PREFIX="${LOG_PREFIX}_${clock}"

printf "Starting job with parameters:\n\n"
echo "N_TRIALS: $N_TRIALS";
echo "N_ENSEMBLES: $N_ENSEMBLES";
echo "MAX_RUNS_PER_GPU: $MAX_RUNS_PER_GPU";
echo "EPOCHS: $EPOCHS";
echo "BATCH_SIZE: $BATCH_SIZE";
echo "DATA_TYPE: $DATA_TYPE";
echo "METRIC: $METRIC";
echo "REG_COEFF: $REG_COEFF";
echo "ATOMISTIC: $ATOMISTIC";
echo "ACTIVE_LEARNING: $ACTIVE_LEARNING";
echo "AL_LOOPS: $AL_LOOPS";
echo "AL_STRATEGY: ${AL_STRATEGY[@]}";
echo "AL_INIT_RATIO: $AL_INIT_RATIO";
echo "LOG_PREFIX: $LOG_PREFIX";
echo "RESTART_INTERVAL: $RESTART_INTERVAL";
echo "METHODS: ${METHODS[@]}";
echo "DATASETS: ${DATASETS[@]}";
echo "EXTRA_ARGS: $EXTRA_ARGS";



#### Find the GPUs and test Test the GPU
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader)
N_GPUS=`echo "$GPUS" | wc -l` #FIXME: convert from str to int
echo "$GPUS"
echo "Found $N_GPUS GPUs. Warming them up..."
for (( i=0; i<$N_GPUS; i++ ))
do
    # Warm up each GPU
    CUDA_VISIBLE_DEVICES=$i python -c "import torch; torch.zeros((1)).to(torch.device('cuda:0'));"
done

# Download the atomistic dataset if it doesn't exist
DATA_ROOT="./data/"
if [ $ATOMISTIC -eq 1 ]; then
  python -c "from schnetpack.datasets import QM9; QM9(\"${DATA_ROOT}/qm9.db\", download=True)"
fi

if [ $ACTIVE_LEARNING -ne 1 ]; then
  AL_STRATEGY=( "none" )
fi

#### Make the ouput log directory
mkdir -p $LOG_PREFIX/.logs
clear

###############################################################################
################################ MAIN JOB LOOP ################################
###############################################################################

#### Main job loop
BASE_ARGS="--save_confidence conf.txt --split_type random --confidence_evaluation_methods cutoff scatter abs_scatter spearman log_likelihood boxplot"
BASE_ARGS="${BASE_ARGS} --batch_size ${BATCH_SIZE}"
for ((i=0; i<$N_TRIALS; i++))   # Repeat for a given number of trials
do
  :
  for dataset in "${DATASETS[@]}"   #
  do
    :

    DATASET_ARGS="--dataset_type $DATA_TYPE"
    if [ $ATOMISTIC -eq 1 ]; then
      DATASET_ARGS="--data_path ${DATA_ROOT}/$dataset.db --atomistic ${DATASET_ARGS}"
      # DATASET_ARGS="${DATASET_ARGS} --init_lr 0.0001 --max_lr 0.0001 --final_lr 0.00001"
      # DATASET_ARGS="${DATASET_ARGS}"
    else
      DATASET_ARGS="--data_path ${DATA_ROOT}/$dataset.csv ${DATASET_ARGS}"
    fi


    for al_strategy in "${AL_STRATEGY[@]}"
    do
      :
      for method in "${METHODS[@]}"
      do
        :

        METHOD_ARGS=""
        NUM_WORKERS=1
        if [ "$method" == "ensemble" ]
        then
          METHOD_ARGS="--ensemble_size $N_ENSEMBLES --threads $N_ENSEMBLES"
          NUM_WORKERS=$N_ENSEMBLES
        fi

        sub_method=$method
        if [ "$method" == "evidence" ]
        then
          METHOD_ARGS="--new_loss --regularizer_coeff $REG_COEFF"
          sub_method="${sub_method}_new_reg_${REG_COEFF}"
        fi

        if [ "$method" == "dropout" ]
        then
          METHOD_ARGS="--ensemble_size $N_ENSEMBLES --dropout 0.2"
        fi

        AL_ARGS="--num_al_loops $AL_LOOPS --al_init_ratio $AL_INIT_RATIO --al_strategy $al_strategy --use_entropy --quiet"

        LOG_ARGS="--save_dir $LOG_PREFIX/$dataset/$sub_method"


        GPU=-1
        while [ $GPU -eq -1 ]
        do
          :
          sleep 1

          for ((j=0; j<$N_GPUS; j++))
          do
            :
            # current_gpu_runs=$(( $(nvidia-smi -i $j | grep python | wc -l) ))
            # current_gpu_runs=$( ps ax | grep "[p]ython active_learning" | awk '{print $1}' | xargs -I{} strings /proc/{}/environ | grep CUDA_VISIBLE_DEVICES | awk -F= '{print $2}' | grep "${j}" | wc -l )
            current_gpu_runs=0
            for pid in `ps ax | grep "[p]ython .*\.py" | awk '{print $1}'`; do
              gpu_ind=$( strings /proc/$pid/environ | grep "CUDA_VISIBLE_DEVICES" | awk -F= '{print $2}' )
              thread_count=$( strings /proc/$pid/environ | (grep "NUM_WORKERS=" || echo "NUM_WORKERS=1") | awk -F= '{print $2}' )
              if [ $gpu_ind -eq $j ]
              then
                (( current_gpu_runs=current_gpu_runs+$thread_count ))
              fi
            done

            if [ $current_gpu_runs -lt $MAX_RUNS_PER_GPU ]
            then
              GPU=$j
              break
            fi
          done
        done



        ENV_VARS="CUDA_VISIBLE_DEVICES=$GPU NUM_WORKERS=$NUM_WORKERS"
        ARGS="--seed $RANDOM --confidence $method --epochs $EPOCHS --metric $METRIC $DATASET_ARGS $METHOD_ARGS $LOG_ARGS $BASE_ARGS $EXTRA_ARGS"
        if [ $ACTIVE_LEARNING -eq 1 ]; then
          # TRAIN ACTIVE LEARNING
          ARGS="$ARGS --use_std"
          CMD="$ENV_VARS python active_learning.py $AL_ARGS $ARGS"
        else
          # TRAIN VANILLA MODEL
          if [ $ATOMISTIC -eq 1 ]; then
            CMD="$ENV_VARS python train_atomistic.py $ARGS"
          else
            CMD="$ENV_VARS python train.py $ARGS"
          fi
          # CMD="CUDA_VISIBLE_DEVICES=$GPU NUM_WORKERS=$NUM_WORKERS python train.py --data_path data/$dataset.csv --confidence $method $EPOCH_ARGS $METHOD_ARGS $LOG_ARGS $BASE_ARGS"
        fi


        # echo
        # echo $CMD
        # # SIMPLE TEST
        # test_code="from time import sleep; import torch; torch.zeros((1)).to(torch.device('cuda:0')); sleep(20); print('test .py done')"
        # CMD="$ENV_VARS python -c \"$test_code\" "

        LOG_FILE=$LOG_PREFIX/.logs/${dataset}_${method}_${al_strategy}_${i}.log
        eval "printf '$CMD\n\n' > $LOG_FILE"
        eval " $CMD &>> $LOG_FILE &"


        # Print the current PID that was just started
        PID=$! #save the PID
        echo "[$GPU/$PID] Started trial $i of $dataset $method. Logs in $LOG_FILE"

        sleep $RESTART_INTERVAL

      done # method
    done # al_strategy
  done # dataset
done # trials

echo "Done queueing all. Now waiting to finish..."
wait
echo "All jobs are done!"
