if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/zero" ]; then
    mkdir ./logs/zero
fi

python ./few.py --config ./config/pro/pro1.yaml --model gpt2 --epochs 200 --name gpt2_pro1_few > logs/few/pro1_few_gpt2.log 2>&1

python ./few.py --config ./config/pro/pro1.yaml --model lstm --epochs 100 --name LSTM_pro1_few > logs/few/pro1_few_lstm.log 2>&1

python ./few.py --config ./config/pro/pro1.yaml --model gru --epochs 100 --name GRU_pro1_few > logs/few/pro1_few_gru.log 2>&1

python ./few.py --config ./config/pro/pro1.yaml --model cnn --epochs 100 --name CNN_pro1_few > logs/few/pro1_few_cnn.log 2>&1
