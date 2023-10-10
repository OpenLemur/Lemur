CUDA_VISIBLE_DEVICES=0
DATA_DIR=data/eval/humaneval

if [ ! -d "$DATA_DIR" ]; then
    echo "Downloading HumanEval data..."
    mkdir -p $DATA_DIR
    wget -P $DATA_DIR https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
fi

PYTHONPATH=$PWD
OUTPUT_DIR=results/humaneval/Llama-2-70b-chat-hf
MODEL=meta-llama/Llama-2-70b-chat-hf

python -m xchat.eval.humaneval.run_eval \
    --data_file $DATA_DIR/HumanEval.jsonl.gz \
    --max_new_token 512 \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 1 \
    --temperature 0.2 \
    --save_dir $OUTPUT_DIR \
    --model $MODEL \
    --tokenizer $MODEL \
    --eval_batch_size 24 \
    --load_in_8bit \
    --chat_format llama2chat
