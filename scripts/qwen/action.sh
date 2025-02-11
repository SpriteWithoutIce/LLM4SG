if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/qwen" ]; then
    mkdir ./logs/qwen
fi

if [ ! -d "./logs/qwen/action" ]; then
    mkdir ./logs/qwen/action
fi

python -m llm.qwen --config ./config/qwen/action/lyh.yaml --epochs 1500 > logs/qwen/action/lyh.log 2>&1
	
python -m llm.qwen --config ./config/qwen/action/yzt1.yaml --epochs 1500 > logs/qwen/action/yzt1.log 2>&1

python -m llm.qwen --config ./config/qwen/action/yzt2.yaml --epochs 1500 > logs/qwen/action/yzt2.log 2>&1

