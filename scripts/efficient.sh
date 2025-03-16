if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/efficient" ]; then
    mkdir ./logs/efficient
fi

if [ ! -d "./logs/efficient/action" ]; then
    mkdir ./logs/efficient/action
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/efficient" ]; then
    mkdir ./logs/efficient
fi

if [ ! -d "./logs/efficient/action-im" ]; then
    mkdir ./logs/efficient/action-im
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/efficient" ]; then
    mkdir ./logs/efficient
fi

if [ ! -d "./logs/efficient/pro" ]; then
    mkdir ./logs/efficient/pro
fi

python run.py --config ./config/action/lyh.yaml --epochs 500 --model efficient --name Efficient_action_lyh_2 > logs/efficient/action/lyh.log 2>&1
	
python run.py --config ./config/action/yzt1.yaml --epochs 500 --model efficient --name Efficient_action_yzt > logs/efficient/action/yzt1.log 2>&1

python run.py --config ./config/action/yzt2.yaml --epochs 500 --model efficient --name Efficient_action_yzt_0815 > logs/efficient/action/yzt2.log 2>&1

python run.py --config ./config/action-im/lyh.yaml --epochs 500 --model efficient --name Efficient_action-im_lyh_im > logs/efficient/action-im/lyh.log 2>&1
	
python run.py --config ./config/action-im/yzt.yaml --epochs 500 --model efficient --name Efficient_action-im_yzt_im > logs/efficient/action-im/yzt.log 2>&1

python run.py --config ./config/pro/pro1.yaml --epochs 500 --model efficient --name Efficient_pro_TENG > logs/efficient/pro/pro1.log 2>&1

python run.py --config ./config/pro/pro2.yaml --epochs 500 --model efficient --name Efficient_pro_2 > logs/efficient/pro/pro2.log 2>&1

python run.py --config ./config/eating.yaml --epochs 500 --model efficient --name Efficient_eating > logs/efficient/eating.log 2>&1

