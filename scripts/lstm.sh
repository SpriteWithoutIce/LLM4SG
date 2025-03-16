if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/lstm" ]; then
    mkdir ./logs/lstm
fi

if [ ! -d "./logs/lstm/action" ]; then
    mkdir ./logs/lstm/action
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/lstm" ]; then
    mkdir ./logs/lstm
fi

if [ ! -d "./logs/lstm/action-im" ]; then
    mkdir ./logs/lstm/action-im
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/lstm" ]; then
    mkdir ./logs/lstm
fi

if [ ! -d "./logs/lstm/pro" ]; then
    mkdir ./logs/lstm/pro
fi

python run.py --config ./config/action/lyh.yaml --epochs 800 --model lstm --name LSTM_action_lyh_2 > logs/lstm/action/lyh.log 2>&1
	
python run.py --config ./config/action/yzt1.yaml --epochs 800 --model lstm --name LSTM_action_yzt > logs/lstm/action/yzt1.log 2>&1

python run.py --config ./config/action/yzt2.yaml --epochs 800 --model lstm --name LSTM_action_yzt_0815 > logs/lstm/action/yzt2.log 2>&1

python run.py --config ./config/action-im/lyh.yaml --epochs 800 --model lstm --name LSTM_action-im_lyh_im > logs/lstm/action-im/lyh.log 2>&1
	
python run.py --config ./config/action-im/yzt.yaml --epochs 800 --model lstm --name LSTM_action-im_yzt_im > logs/lstm/action-im/yzt.log 2>&1

python run.py --config ./config/action-im/zxy.yaml --epochs 800 --model lstm --name LSTM_action-im_zxy > logs/lstm/action-im/zxy.log 2>&1

python run.py --config ./config/action-im/zxy2.yaml --epochs 800 --model lstm --name LSTM_action-im_zxy_2 > logs/lstm/action-im/zxy2.log 2>&1

python run.py --config ./config/pro/pro1.yaml --epochs 800 --model lstm --name LSTM_pro_TENG > logs/lstm/pro/pro1.log 2>&1

python run.py --config ./config/pro/pro2.yaml --epochs 800 --model lstm --name LSTM_pro_2 > logs/lstm/pro/pro2.log 2>&1

python run.py --config ./config/eating.yaml --epochs 800 --model lstm --name LSTM_eating > logs/lstm/eating.log 2>&1

