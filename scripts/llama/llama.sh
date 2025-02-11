if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/llama" ]; then
    mkdir ./logs/llama
fi

python ./llm/llama.py --config ./config/llama/hospital.yaml > logs/llama/hospital.log 2>&1
	
python ./llm/llama.py --config ./config/llama/yzt_im.yaml > logs/llama/yzt_im.log 2>&1

python ./llm/llama.py --config ./config/llama/zxy_im.yaml > logs/llama/zxy_im.log 2>&1

python ./llm/llama.py --config ./config/llama/zxy_im2.yaml > logs/llama/zxy_im2.log 2>&1

