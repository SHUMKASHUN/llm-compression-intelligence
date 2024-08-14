
# CUDA_VISIBLE_DEVICES=0 python -u main.py\
#     --model_name huggyllama/llama-7b\
#     --task_name cc\
#     --block_size 1900\
#     --stride 512\
#     --batch_size 1\
#     --cluster 0 \
#     --flash 
export HF_ENDPOINT=https://hf-mirror.com
model=$1
device=$2
begin=$3
end=$4

for i in $( eval echo {$begin..$end} )
do
CUDA_VISIBLE_DEVICES=$device python -u main.py\
    --model_name $model\
    --task_name cc\
    --block_size 1900\
    --stride 512 \
    --batch_size 1\
    --cluster $i \
    --flash
done