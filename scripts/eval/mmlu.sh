export CUDA_VISIBLE_DEVICES=4,5,6,7


DATA_DIR=data/eval/mmlu

if [ ! -d $DATA_DIR ]; then
    echo "Downloading MMLU data..."
    wget -O data/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
    mkdir -p data/eval/mmlu_data
    tar -xvf data/mmlu_data.tar -C data/eval/mmlu_data
    mv data/eval/mmlu_data/data $DATA_DIR && rm -r data/eval/mmlu_data data/mmlu_data.tar
fi

MODEL_DIR=meta-llama/Llama-2-7b-hf
OUTPUT_DIR=results/mmlu/llama-2-7b-hf
python -m xchat.eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir $DATA_DIR \
    --save_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_DIR \
    --tokenizer_name_or_path $MODEL_DIR \
    --eval_batch_size 4

# MODEL_DIR=bigcode/starcoder
# OUTPUT_DIR=results/mmlu/starcoder
# python -m xchat.eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir $DATA_DIR \
#     --save_dir $OUTPUT_DIR \
#     --model_name_or_path $MODEL_DIR \
#     --tokenizer_name_or_path $MODEL_DIR \
#     --eval_batch_size 16
