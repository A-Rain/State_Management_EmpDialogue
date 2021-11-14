if [ ! $# -eq 8 ]; then
    echo "need 8 parameters"
    exit
fi

BERT_PATH=$1
LENGTH=$2
lr=$3
bsz=$4
epo=$5
seed=$6
gpu_id=$7
warmup=$8

WORK_DIR=$(cd $(dirname $(dirname $0));pwd)

cd ${WORK_DIR}/preprocess

CACHE_DIR=${WORK_DIR}/cache/goEmotion
mkdir -p ${CACHE_DIR}

if [ "`ls -A ${CACHE_DIR}`" = "" ]; then
    echo "build/rebiuld the cache goEmotion file"
    python build_GoEmotion_input_format.py \
        --bert_path ${BERT_PATH} \
        --data_dir ${WORK_DIR}/data/goEmotion \
        --cache_dir ${CACHE_DIR} \
        --max_length ${LENGTH}
else
    echo "already have the cache goEmotion file"
fi

SAVE_DIR=${WORK_DIR}/save/goEmotion
mkdir -p ${SAVE_DIR}
MODEL_FILE=${SAVE_DIR}/best.pt

if [ ! -f ${MODEL_FILE} ]; then
    echo "train emotion classifier"
    python train_emotion_classifier.py \
        --bert_path ${BERT_PATH} \
        --cache_dir ${CACHE_DIR} \
        --save_dir ${SAVE_DIR} \
        --do_train \
        --do_eval \
        --use_gpu \
        --gpu_id 0 \
        --seed ${seed} \
        --lr ${lr} \
        --train_batch_size ${bsz} \
        --eval_batch_size ${bsz} \
        --num_train_epochs ${epo} \
        --warmup ${warmup} \
        --max_length ${LENGTH}
else
    echo "emotion classifier already exists"
fi

python emotion_annotate.py \
    --bert_path ${BERT_PATH} \
    --save_dir ${SAVE_DIR} \
    --data_dir ${WORK_DIR}/data/empathetic \
    --use_gpu \
    --gpu_id 2 \
    --max_length ${LENGTH}




