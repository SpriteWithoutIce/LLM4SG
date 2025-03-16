if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/gru" ]; then
    mkdir ./logs/gru
fi

if [ ! -d "./logs/gru/action" ]; then
    mkdir ./logs/gru/action
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/gru" ]; then
    mkdir ./logs/gru
fi

if [ ! -d "./logs/gru/action-im" ]; then
    mkdir ./logs/gru/action-im
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/gru" ]; then
    mkdir ./logs/gru
fi

if [ ! -d "./logs/gru/pro" ]; then
    mkdir ./logs/gru/pro
fi

python run.py --config ./config/action/lyh.yaml --epochs 800 --model gru --name GRU_action_lyh_2 > logs/gru/action/lyh.log 2>&1
	
python run.py --config ./config/action/yzt1.yaml --epochs 800 --model gru --name GRU_action_yzt > logs/gru/action/yzt1.log 2>&1

python run.py --config ./config/action/yzt2.yaml --epochs 800 --model gru --name GRU_action_yzt_0815 > logs/gru/action/yzt2.log 2>&1

python run.py --config ./config/action-im/lyh.yaml --epochs 800 --model gru --name GRU_action-im_lyh_im > logs/gru/action-im/lyh.log 2>&1
	
python run.py --config ./config/action-im/yzt.yaml --epochs 800 --model gru --name GRU_action-im_yzt_im > logs/gru/action-im/yzt.log 2>&1

python run.py --config ./config/pro/pro1.yaml --epochs 800 --model gru --name GRU_pro_TENG > logs/gru/pro/pro1.log 2>&1

python run.py --config ./config/pro/pro2.yaml --epochs 800 --model gru --name GRU_pro_2 > logs/gru/pro/pro2.log 2>&1

python run.py --config ./config/eating.yaml --epochs 800 --model gru --name GRU_eating > logs/gru/eating.log 2>&1

