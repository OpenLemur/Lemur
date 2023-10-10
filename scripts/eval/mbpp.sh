export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_DIR=data/eval/mbpp
OUTPUT_DIR=results/mbpp/Llama-2-70b-chat-hf
MODEL_PATH=meta-llama/Llama-2-70b-chat-hf
python -m xchat.eval.mbpp.run_eval \
    --data_dir $DATA_DIR \
    --max_new_token 650 \
    --max_num_examples 500 \
    --save_dir $OUTPUT_DIR \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --eval_batch_size 8 \
    --chat_format llama2chat \
    --load_in_8bit \
    --greedy_decoding \
    --few_shot \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 1
