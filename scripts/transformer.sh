if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/transformer" ]; then
    mkdir ./logs/transformer
fi

if [ ! -d "./logs/transformer/action" ]; then
    mkdir ./logs/transformer/action
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/transformer" ]; then
    mkdir ./logs/transformer
fi

if [ ! -d "./logs/transformer/action-im" ]; then
    mkdir ./logs/transformer/action-im
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/transformer" ]; then
    mkdir ./logs/transformer
fi

if [ ! -d "./logs/transformer/pro" ]; then
    mkdir ./logs/transformer/pro
fi

python ./patchTST.py --config ./config/action/lyh.yaml > logs/ablation.log 2>&1

python ./patchTST.py --config ./config/action/yzt1.yaml >> logs/ablation.log 2>&1

python ./patchTST.py --config ./config/action/yzt2.yaml >> logs/ablation.log 2>&1

python ./patchTST.py --config ./config/action-im/lyh.yaml >> logs/ablation.log 2>&1

python ./patchTST.py --config ./config/action-im/yzt.yaml >> logs/ablation.log 2>&1

python ./patchTST.py --config ./config/pro/pro1.yaml >> logs/ablation.log 2>&1

python ./patchTST.py --config ./config/pro/pro2.yaml >> logs/ablation.log 2>&1

python ./patchTST.py --config ./config/eating.yaml >> logs/ablation.log 2>&1