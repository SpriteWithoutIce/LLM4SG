if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/zero" ]; then
    mkdir ./logs/zero
fi

python ./few.py --config ./config/action/yzt2.yaml --model gpt2 --epochs 200 --name gpt2_yzt2_few > logs/few/yzt2_few_gpt2.log 2>&1

python ./few.py --config ./config/action/yzt2.yaml --model lstm --epochs 100 --name LSTM_yzt2_few > logs/few/yzt2_few_lstm.log 2>&1

python ./few.py --config ./config/action/yzt2.yaml --model gru --epochs 100 --name GRU_yzt2_few > logs/few/yzt2_few_gru.log 2>&1

python ./few.py --config ./config/action/yzt2.yaml --model cnn --epochs 100 --name CNN_yzt2_few > logs/few/yzt2_few_cnn.log 2>&1
