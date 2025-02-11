if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/llama" ]; then
    mkdir ./logs/llama
fi

if [ ! -d "./logs/llama/action-im" ]; then
    mkdir ./logs/llama/action-im
fi

python -m llm.llama --config ./config/llama/action-im/lyh.yaml --epochs 1500 > logs/llama/action-im/lyh.log 2>&1
	
python -m llm.llama --config ./config/llama/action-im/yzt.yaml --epochs 1500 > logs/llama/action-im/yzt.log 2>&1

python -m llm.llama --config ./config/llama/action-im/zxy.yaml --epochs 1500 > logs/llama/action-im/zxy.log 2>&1

python -m llm.llama --config ./config/llama/action-im/zxy2.yaml --epochs 1500 > logs/llama/action-im/zxy2.log 2>&1



