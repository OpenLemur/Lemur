export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR=data/eval/bbh

if [ ! -d "data" ]; then
    mkdir data
fi

if [ ! -d "data/eval" ]; then
    mkdir data/eval
fi

if [ ! -d $DATA_DIR ]; then
    echo "Downloading BBH data..."
    mkdir -p data/downloads
    wget -O data/downloads/bbh_data.zip https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip
    unzip data/downloads/bbh_data.zip -d data/downloads/bbh
    mv data/downloads/bbh/BIG-Bench-Hard-main data/eval/bbh && rm -rf data/downloads/
fi

MODEL_DIR=codellama/CodeLlama-7b-Instruct-hf
OUTPUT_DIR=results/bbh/llama-2-7b-hf

python -m xchat.eval.bbh.run_eval \
    --data_dir data/eval/bbh/ \
    --save_dir $OUTPUT_DIR \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR \
    --eval_batch_size 20 \
    --load_in_8bit \
    --no_cot \
    --chat_format codellama-instruct
