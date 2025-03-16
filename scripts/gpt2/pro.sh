if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/gpt2" ]; then
    mkdir ./logs/gpt2
fi

if [ ! -d "./logs/gpt2/pro" ]; then
    mkdir ./logs/gpt2/pro
fi

python ./llm/gpt2.py --config ./config/gpt2/pro/pro1.yaml --epochs 500 > logs/gpt2/pro1.log 2>&1
	
python ./llm/gpt2.py --config ./config/gpt2/pro/pro2.yaml --epochs 500 > logs/gpt2/pro2.log 2>&1

