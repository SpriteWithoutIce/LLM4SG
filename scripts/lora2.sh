python ./run.py --config ./config/action/yzt1.yaml --model qwen --epochs 500 --name qwen_yzt1_noLSTM --lstm False > logs/qwen/action/yzt1_noLSTM.log 2>&1

python ./run.py --config ./config/action/yzt2.yaml --model qwen --epochs 500 --name qwen_yzt2_noLSTM --lstm False > logs/qwen/action/yzt2_noLSTM.log 2>&1