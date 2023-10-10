N_GPUS=4
gpus='"device=0,1,2,3"'
MODEL_PATH=OpenLemur/lemur-70b-v1
MODEL_NAME=lemur-70b-v1
# HF_HOME=""

docker run --gpus $gpus --rm \
    -v $HF_HOME:/root/.cache/huggingface \
    --shm-size=10.24gb \
    --name vllm-$MODEL_NAME \
    ranpox/fastchat:lemur \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size $N_GPUS \
    --served-model-name $MODEL_NAME \
    --max-num-batched-tokens 4096 \
    --load-format pt \
    --port 8000
