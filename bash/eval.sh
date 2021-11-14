usage() {
  echo "Usage: ${0} [--hyp] [--ref] [--out] [--bert] [--mode] [--gpu_id]" 1>&2
  exit 1 
}
while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    --hyp)
        HYP=${2}
        shift 2
        ;;
    --ref)
        REF=${2}
        shift 2
        ;;
    --out)
        OUT=${2}
        shift 2
        ;;
    --bert)
        BERT=${2}
        shift 2
        ;;
    --mode)
        MODE=${2}
        shift 2
        ;;
    --gpu_id)
        GPU=${2}
        shift 2
        ;;
    *)
      usage
      shift
      ;;
  esac
done


WORK_DIR=$(cd $(dirname $(dirname $0));pwd)
cd ${WORK_DIR}

if [ "${MODE}" == "LM" ]; then
    echo "evaluate LM-based model"
    CUDA_VISIBLE_DEVICES=${GPU} python metric_utils/metric.py \
        --hyp ${HYP} \
        --ref ${REF} \
        --output ${OUT} \
        --BERT_path ${BERT} \
        --rescale \
        --lower
else
    echo "evaluate Trs-based model"
    CUDA_VISIBLE_DEVICES=${GPU} python metric_utils/metric.py \
        --hyp ${HYP} \
        --ref ${REF} \
        --output ${OUT} \
        --BERT_path ${BERT} \
        --rescale \
        --space_token \
        --lower
fi