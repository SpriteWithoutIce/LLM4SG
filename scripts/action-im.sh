python -m llm.gpt2 --config ./config/gpt2/eating.yaml --epochs 500 > logs/gpt2/eating.log 2>&1

python -m llm.gpt2 --config ./config/gpt2/hospital.yaml --epochs 500 > logs/gpt2/hospital.log 2>&1

python -m llm.llama --config ./config/llama/eating.yaml --epochs 500 > logs/llama/eating.log 2>&1

python -m llm.llama --config ./config/llama/hospital.yaml --epochs 500 > logs/llama/hospital.log 2>&1

python -m llm.qwen --config ./config/qwen/eating.yaml --epochs 500 > logs/qwen/eating.log 2>&1

python -m llm.qwen --config ./config/qwen/hospital.yaml --epochs 500 > logs/qwen/hospital.log 2>&1
