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

CACHE_DIR=${WORK_DIR}/cache/intent
mkdir -p ${CACHE_DIR}

if [ "`ls -A ${CACHE_DIR}`" = "" ]; then
    echo "build/rebiuld the cache intent file"
    python build_intent_input_format.py \
        --bert_path ${BERT_PATH} \
        --data_dir ${WORK_DIR}/data/intent \
        --cache_dir ${CACHE_DIR} \
        --max_length ${LENGTH}
else
    echo "already have the cache intent file"
fi

SAVE_DIR=${WORK_DIR}/save/intent
mkdir -p ${SAVE_DIR}
MODEL_FILE=${SAVE_DIR}/best.pt

if [ ! -f ${MODEL_FILE} ]; then
    echo "train intent classifier"
    python train_intent_classifier.py \
        --bert_path ${BERT_PATH} \
        --cache_dir ${CACHE_DIR} \
        --save_dir ${SAVE_DIR} \
        --do_train \
        --do_eval \
        --use_gpu \
        --gpu_id 2 \
        --seed ${seed} \
        --lr ${lr} \
        --train_batch_size ${bsz} \
        --eval_batch_size ${bsz} \
        --num_train_epochs ${epo} \
        --warmup ${warmup} \
        --max_length ${LENGTH}
else
    echo "intent classifier already exists"
fi

python intent_annotate.py \
    --bert_path ${BERT_PATH} \
    --save_dir ${SAVE_DIR} \
    --data_dir ${WORK_DIR}/data/empathetic \
    --use_gpu \
    --gpu_id 2 \
    --max_length ${LENGTH} \
    --max_sent_num 3