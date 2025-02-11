if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/qwen" ]; then
    mkdir ./logs/qwen
fi

if [ ! -d "./logs/qwen/action-im" ]; then
    mkdir ./logs/qwen/action-im
fi

python -m llm.qwen --config ./config/qwen/action-im/lyh.yaml --epochs 1500 > logs/qwen/action-im/lyh.log 2>&1
	
python -m llm.qwen --config ./config/qwen/action-im/yzt.yaml --epochs 1500 > logs/qwen/action-im/yzt.log 2>&1

python -m llm.qwen --config ./config/qwen/action-im/zxy.yaml --epochs 1500 > logs/qwen/action-im/zxy.log 2>&1

python -m llm.qwen --config ./config/qwen/action-im/zxy2.yaml --epochs 1500 > logs/qwen/action-im/zxy2.log 2>&1



