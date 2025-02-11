if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/qwen" ]; then
    mkdir ./logs/qwen
fi

python ./llm/qwen.py --config ./config/qwen/hospital.yaml > logs/qwen/hospital.log 2>&1
	
python ./llm/qwen.py --config ./config/qwen/yzt_im.yaml > logs/qwen/yzt_im.log 2>&1

python ./llm/qwen.py --config ./config/qwen/zxy_im.yaml > logs/qwen/zxy_im.log 2>&1

python ./llm/qwen.py --config ./config/qwen/zxy_im2.yaml > logs/qwen/zxy_im2.log 2>&1

