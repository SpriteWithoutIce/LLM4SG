if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/llama" ]; then
    mkdir ./logs/llama
fi

if [ ! -d "./logs/llama/pro" ]; then
    mkdir ./logs/llama/pro
fi

python ./llm/llama.py --config ./config/llama/pro/pro1.yaml --epochs 500 > logs/llama/pro1.log 2>&1
	
python ./llm/llama.py --config ./config/llama/pro/pro2.yaml --epochs 500 > logs/llama/pro2.log 2>&1

