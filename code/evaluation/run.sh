
CUDA_VISIBLE_DEVICES=0 python -u main.py\
    --model_name huggyllama/llama-7b\
    --task_name cc\
    --block_size 1900\
    --stride 512\
    --batch_size 8\
    --cluster 0 \
    --flash 
