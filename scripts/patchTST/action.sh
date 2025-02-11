if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/patchTST" ]; then
    mkdir ./logs/patchTST
fi

if [ ! -d "./logs/patchTST/action" ]; then
    mkdir ./logs/patchTST/action
fi
	
python ./patchTST.py --config ./config/patchTST/action/lyh.yaml --epochs 300 > logs/patchTST/action/lyh.log 2>&1

python ./patchTST.py --config ./config/patchTST/action/yzt1.yaml --epochs 300 > logs/patchTST/action/yzt1.log 2>&1

python ./patchTST.py --config ./config/patchTST/action/yzt2.yaml --epochs 300 > logs/patchTST/action/yzt2.log 2>&1

