python Qwen1_5MoE.py '/data1/ldk/model/Qwen1.5' c4

python Qwen1_5MoE.py '/data1/ldk/model/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9' c4 --sparsity .75 --save '/data1/ldk/SPNN/sparsegpt' --gpu_ids '3,4,5,6,7'
