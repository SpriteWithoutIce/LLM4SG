if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few" ]; then
    mkdir ./logs/few
fi

python ./few.py --config ./config/action/lyh.yaml --model lstm --epochs 100 --name LSTM_lyh_few > logs/few/lyh_few_lstm.log 2>&1

python ./few.py --config ./config/action/lyh.yaml --model gru --epochs 100 --name GRU_lyh_few > logs/few/lyh_few_gru.log 2>&1

python ./few.py --config ./config/action/lyh.yaml --model cnn --epochs 100 --name CNN_lyh_few > logs/few/lyh_few_cnn.log 2>&1
