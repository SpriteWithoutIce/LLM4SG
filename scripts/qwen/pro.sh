if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/qwen" ]; then
    mkdir ./logs/qwen
fi

if [ ! -d "./logs/qwen/pro" ]; then
    mkdir ./logs/qwen/pro
fi

python ./llm/qwen.py --config ./config/qwen/pro/pro1.yaml --epochs 500 > logs/qwen/pro1.log 2>&1
	
python ./llm/qwen.py --config ./config/qwen/pro/pro2.yaml --epochs 500 > logs/qwen/pro2.log 2>&1

