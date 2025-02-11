if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/GPT2" ]; then
    mkdir ./logs/GPT2
fi

if [ ! -d "./logs/GPT2/action" ]; then
    mkdir ./logs/GPT2/action
fi
	
python ./llm/gpt2.py --config ./config/gpt2/action/yzt1.yaml --epochs 1500 > logs/GPT2/action/yzt1.log 2>&1

python ./llm/gpt2.py --config ./config/gpt2/action/yzt2.yaml --epochs 1500 > logs/GPT2/action/yzt2.log 2>&1

