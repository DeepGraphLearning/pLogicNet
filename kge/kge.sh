#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=kge
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATA_PATH=$3
GPU_DEVICE=$4

#Only used in training
BATCH_SIZE=$5
NEGATIVE_SAMPLE_SIZE=$6
HIDDEN_DIM=$7
GAMMA=$8
ALPHA=$9
LEARNING_RATE=${10}
MAX_STEPS=${11}
TEST_BATCH_SIZE=${12}
WORKSPACE_PATH=${13}
TOP_K=${14}

SAVE=$WORKSPACE_PATH/"$MODEL"

if [ $MODE == "train" ]
then

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv --record --valid_steps 50000 \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE --workspace_path $WORKSPACE_PATH --topk $TOP_K \
    ${15} ${16} ${17} ${18} ${19} ${20} ${21}


elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE
    
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

else
   echo "Unknown MODE" $MODE
fi
