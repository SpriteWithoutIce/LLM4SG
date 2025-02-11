if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/patchTST" ]; then
    mkdir ./logs/patchTST
fi

if [ ! -d "./logs/patchTST/action-im" ]; then
    mkdir ./logs/patchTST/action-im
fi

python ./patchTST.py --config ./config/patchTST/action-im/lyh.yaml > logs/patchTST/action-im/lyh.log 2>&1
	
python ./patchTST.py --config ./config/patchTST/action-im/yzt.yaml > logs/patchTST/action-im/yzt.log 2>&1

python ./patchTST.py --config ./config/patchTST/action-im/zxy.yaml > logs/patchTST/action-im/zxy.log 2>&1

python ./patchTST.py --config ./config/patchTST/action-im/zxy2.yaml > logs/patchTST/action-im/zxy2.log 2>&1



