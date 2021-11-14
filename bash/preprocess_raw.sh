WORK_DIR=$(cd $(dirname $(dirname $0));pwd)

cd ${WORK_DIR}/preprocess

OUTPUT_DIR=${WORK_DIR}/data/empathetic
mkdir -p ${OUTPUT_DIR}

python ./convert_csv_into_json.py \
    --raw_dir ${WORK_DIR}/data/empathetic_raw \
    --output_dir ${OUTPUT_DIR}
