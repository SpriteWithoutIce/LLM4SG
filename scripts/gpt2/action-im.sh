if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/gpt2" ]; then
    mkdir ./logs/gpt2
fi

if [ ! -d "./logs/gpt2/action-im" ]; then
    mkdir ./logs/gpt2/action-im
fi

python -m llm.gpt2 --config ./config/gpt2/action-im/lyh.yaml --epochs 1500 > logs/gpt2/action-im/lyh.log 2>&1
	
python -m llm.gpt2 --config ./config/gpt2/action-im/yzt.yaml --epochs 1500 > logs/gpt2/action-im/yzt.log 2>&1

python -m llm.gpt2 --config ./config/gpt2/action-im/zxy.yaml --epochs 1500 > logs/gpt2/action-im/zxy.log 2>&1

python -m llm.gpt2 --config ./config/gpt2/action-im/zxy2.yaml --epochs 1500 > logs/gpt2/action-im/zxy2.log 2>&1



