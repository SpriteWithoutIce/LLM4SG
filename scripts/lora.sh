python ./run.py --config ./config/action/yzt1.yaml --model qwen --epochs 500 --name qwen_yzt1_noLora --lora False > logs/qwen/action/yzt1_noLora.log 2>&1

python ./run.py --config ./config/action/yzt2.yaml --model qwen --epochs 500 --name qwen_yzt2_noLora --lora False > logs/qwen/action/yzt2_noLora.log 2>&1