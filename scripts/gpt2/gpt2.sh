if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/GPT2" ]; then
    mkdir ./logs/GPT2
fi

python ./llm/gpt2.py --config ./config/gpt2/hospital.yaml --epochs 3000 > logs/GPT2/hospital.log 2>&1
	
python ./llm/gpt2.py --config ./config/gpt2/yzt_im.yaml --epochs 3000 > logs/GPT2/yzt_im.log 2>&1

python ./llm/gpt2.py --config ./config/gpt2/zxy_im.yaml --epochs 3000 > logs/GPT2/zxy_im.log 2>&1

python ./llm/gpt2.py --config ./config/gpt2/zxy_im2.yaml --epochs 3000 > logs/GPT2/zxy_im2.log 2>&1

