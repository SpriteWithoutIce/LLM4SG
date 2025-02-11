if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/llama" ]; then
    mkdir ./logs/llama
fi

if [ ! -d "./logs/llama/action" ]; then
    mkdir ./logs/llama/action
fi

python ./llm/llama.py --config ./config/llama/action/lyh.yaml --epochs 1500 > logs/llama/action/lyh.log 2>&1
	
python ./llm/llama.py --config ./config/llama/action/yzt1.yaml --epochs 1500 > logs/llama/action/yzt1.log 2>&1

python ./llm/llama.py --config ./config/llama/action/yzt2.yaml --epochs 1500 > logs/llama/action/yzt2.log 2>&1

