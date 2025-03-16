if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/cnn" ]; then
    mkdir ./logs/cnn
fi

if [ ! -d "./logs/cnn/action" ]; then
    mkdir ./logs/cnn/action
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/cnn" ]; then
    mkdir ./logs/cnn
fi

if [ ! -d "./logs/cnn/action-im" ]; then
    mkdir ./logs/cnn/action-im
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/cnn" ]; then
    mkdir ./logs/cnn
fi

if [ ! -d "./logs/cnn/pro" ]; then
    mkdir ./logs/cnn/pro
fi

python run.py --config ./config/action/lyh.yaml --epochs 200 --model cnn --name CNN_action_lyh_2 > logs/cnn/action/lyh.log 2>&1
	
python run.py --config ./config/action/yzt1.yaml --epochs 200 --model cnn --name CNN_action_yzt > logs/cnn/action/yzt1.log 2>&1

python run.py --config ./config/action/yzt2.yaml --epochs 200 --model cnn --name CNN_action_yzt_0815 > logs/cnn/action/yzt2.log 2>&1

python run.py --config ./config/action-im/lyh.yaml --epochs 200 --model cnn --name CNN_action-im_lyh_im > logs/cnn/action-im/lyh.log 2>&1
	
python run.py --config ./config/action-im/yzt.yaml --epochs 200 --model cnn --name CNN_action-im_yzt_im > logs/cnn/action-im/yzt.log 2>&1

python run.py --config ./config/pro/pro1.yaml --epochs 200 --model cnn --name CNN_pro_TENG > logs/cnn/pro/pro1.log 2>&1

python run.py --config ./config/pro/pro2.yaml --epochs 200 --model cnn --name CNN_pro_2 > logs/cnn/pro/pro2.log 2>&1

python run.py --config ./config/eating.yaml --epochs 200 --model cnn --name CNN_eating > logs/cnn/eating.log 2>&1

