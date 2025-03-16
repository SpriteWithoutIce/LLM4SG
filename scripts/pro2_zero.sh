if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/zero" ]; then
    mkdir ./logs/zero
fi

python ./few.py --config ./config/eating.yaml --model gpt2 --epochs 200 --name gpt2_eating_few > logs/few/eating_few_gpt2.log 2>&1

python ./few.py --config ./config/eating.yaml --model lstm --epochs 100 --name LSTM_eating_few > logs/few/eating_few_lstm.log 2>&1

python ./few.py --config ./config/eating.yaml --model gru --epochs 100 --name GRU_eating_few > logs/few/eating_few_gru.log 2>&1

python ./few.py --config ./config/eating.yaml --model cnn --epochs 100 --name CNN_eating_few > logs/few/eating_few_cnn.log 2>&1
