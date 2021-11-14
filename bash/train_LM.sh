usage() {
  echo "Usage: ${0} [--gpt_path] [--bert_path] [--gpu_id] [--epoch] [--lr_NLU] [--lr_NLG] [--bsz_NLU] [--bsz_NLG]" 1>&2
  exit 1 
}
while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    --gpt_path)
        GPT=${2}
        shift 2
        ;;
    --bert_path)
        BERT=${2}
        shift 2
        ;;
    --gpu_id)
        GPU=${2}
        shift 2
        ;;
    --epoch)
        EPO=${2}
        shift 2
        ;;
    --lr_NLU)
        LRU=${2}
        shift 2
        ;;
    --lr_NLG)
        LRG=${2}
        shift 2
        ;;
    --bsz_NLU)
        BSZU=${2}
        shift 2
        ;;
    --bsz_NLG)
        BSZG=${2}
        shift 2
        ;;
    *)
      usage
      shift
      ;;
  esac
done


WORK_DIR=$(cd $(dirname $(dirname $0));pwd)

cd ${WORK_DIR}/LM_exp

DATA_DIR=${WORK_DIR}/data/empathetic
CACHE_DIR=${WORK_DIR}/cache/LM_exp
mkdir -p ${CACHE_DIR}

if [ "`ls -A ${CACHE_DIR}`" = "" ]; then
    echo "build/rebiuld the cache LM_exp file"
    python preprocess.py \
        --gpt_path ${GPT} \
        --bert_path ${BERT} \
        --cache_dir ${CACHE_DIR} \
        --data_dir ${DATA_DIR} \
        --max_length 128
else
    echo "already have the cache LM_exp file"
fi

SAVE_DIR=${WORK_DIR}/save/LM_exp
mkdir -p ${SAVE_DIR}

python train.py \
    --gpt_path ${GPT} \
    --bert_path ${BERT} \
    --data_dir ${DATA_DIR} \
    --cache_dir ${CACHE_DIR} \
    --save_dir ${SAVE_DIR} \
    --do_eval \
    --use_gpu \
    --gpu_id ${GPU} \
    --train_batch_size_NLU ${BSZU} \
    --lr_NLU ${LRU} \
    --train_batch_size_NLG ${BSZG} \
    --lr_NLG ${LRG} \
    --alternate_num_epochs ${EPO}


