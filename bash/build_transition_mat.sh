WORK_DIR=$(cd $(dirname $(dirname $0));pwd)

cd ${WORK_DIR}/preprocess

python build_transition_matrix.py \
    --data_dir ${WORK_DIR}/data/empathetic