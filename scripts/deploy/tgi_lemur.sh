model=OpenLemur/lemur-70b-v1
name="lemur-70b-v1"
# HF_HOME=""
volume=$HF_HOME/hub
gpus='"device=0,1,2,3"'
num_shard=4
dtype="bfloat16"
quantize="bitsandbytes"
max_input_length=4095
max_total_tokens=4096
max_batch_prefill_tokens=16380
max_batch_total_tokens=16384

docker run --gpus $gpus \
    --shm-size 1g --rm \
    -p 8080:80 -v $volume:/data \
    --name tgi-$name \
    ghcr.io/huggingface/text-generation-inference:1.0.3 \
    --model-id $model \
    --sharded false \
    --num-shard $num_shard \
    --dtype $dtype \
    --max-input-length $max_input_length \
    --max-total-tokens $max_total_tokens \
    --max-batch-prefill-tokens $max_batch_prefill_tokens \
    --max-batch-total-tokens $max_batch_total_tokens \
    --max-stop-sequences 10
