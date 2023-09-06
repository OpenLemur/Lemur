export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_DIR=data/eval/gsm

if [ ! -d $DATA_DIR ]; then
    echo "Downloading GSM data..."
    mkdir -p $DATA_DIR
    wget -P $DATA_DIR https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl
fi


MODEL_DIR=meta-llama/Llama-2-7b-hf
OUTPUT_DIR=results/gsm/llama-2-7b-hf
python -m xchat.eval.gsm.run_eval \
    --data_dir $DATA_DIR \
    --max_num_examples 32 \
    --save_dir $OUTPUT_DIR \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR \
    --eval_batch_size 16 \
    --n_shot 8 \
    --load_in_8bit

# MODEL_DIR=OpenLemur/lemur-70b-v1
# OUTPUT_DIR=results/gsm/lemur-70b-v1
# python -m xchat.eval.gsm.run_eval \
#     --data_dir $DATA_DIR \
#     --max_num_examples 32 \
#     --save_dir $OUTPUT_DIR \
#     --model $MODEL_DIR \
#     --tokenizer $MODEL_DIR \
#     --eval_batch_size 16 \
#     --n_shot 8 \
#     --load_in_8bit

# MODEL_DIR=lmsys/vicuna-13b-v1.5
# OUTPUT_DIR=results/gsm/vicuna-13b-v1.5
# python -m xchat.eval.gsm.run_eval \
#     --data_dir $DATA_DIR \
#     --max_num_examples 32 \
#     --save_dir $OUTPUT_DIR \
#     --model $MODEL_DIR \
#     --tokenizer $MODEL_DIR \
#     --eval_batch_size 16 \
#     --n_shot 8 \
#     --chat_format vicuna \
#     --load_in_8bit

# MODEL_DIR=codellama/CodeLlama-34b-Instruct-hf
# OUTPUT_DIR=results/gsm/codellama-34b-instruct-hf
# python -m xchat.eval.gsm.run_eval \
#     --data_dir $DATA_DIR \
#     --save_dir $OUTPUT_DIR \
#     --model $MODEL_DIR \
#     --tokenizer $MODEL_DIR \
#     --eval_batch_size 48 \
#     --n_shot 8 \
#     --chat_format codellama-instruct
