if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/resnet" ]; then
    mkdir ./logs/resnet
fi

if [ ! -d "./logs/resnet/action" ]; then
    mkdir ./logs/resnet/action
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/resnet" ]; then
    mkdir ./logs/resnet
fi

if [ ! -d "./logs/resnet/action-im" ]; then
    mkdir ./logs/resnet/action-im
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/resnet" ]; then
    mkdir ./logs/resnet
fi

if [ ! -d "./logs/resnet/pro" ]; then
    mkdir ./logs/resnet/pro
fi

python run.py --config ./config/action/lyh.yaml --epochs 500 --model resnet --name Resnet_action_lyh_2 > logs/resnet/action/lyh.log 2>&1
	
python run.py --config ./config/action/yzt1.yaml --epochs 500 --model resnet --name Resnet_action_yzt > logs/resnet/action/yzt1.log 2>&1

python run.py --config ./config/action/yzt2.yaml --epochs 500 --model resnet --name Resnet_action_yzt_0815 > logs/resnet/action/yzt2.log 2>&1

python run.py --config ./config/action-im/lyh.yaml --epochs 500 --model resnet --name Resnet_action-im_lyh_im > logs/resnet/action-im/lyh.log 2>&1
	
python run.py --config ./config/action-im/yzt.yaml --epochs 500 --model resnet --name Resnet_action-im_yzt_im > logs/resnet/action-im/yzt.log 2>&1

python run.py --config ./config/pro/pro1.yaml --epochs 500 --model resnet --name Resnet_pro_TENG > logs/resnet/pro/pro1.log 2>&1

python run.py --config ./config/pro/pro2.yaml --epochs 500 --model resnet --name Resnet_pro_2 > logs/resnet/pro/pro2.log 2>&1

python run.py --config ./config/eating.yaml --epochs 500 --model resnet --name Resnet_eating > logs/resnet/eating.log 2>&1

