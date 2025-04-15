# LLM4SC

## Getting Started

### Code Explanation

* `answer`: for each task, the classification results will be saved here
* `config`: for each task and model, the training configurations are saved here, including all four tasks and its configurations, all yaml files.
* `data/dataset`: datasets for all tasks
* `exp_ML`: experiments for Machine Learning Models.
* `llm`: old version for llms training(gpt2, qwen, llama)
* `models`: all models testing in the paper, including RNN-based(LSTM, GRU), CNN-based(three-layer CNN, ResNet), transformer(vanilla-transformer), llm(gpt2, qwen, llama)
* `scripts`: bash scripts for instructions
* `utils`: embedding layers

* `few.py`: Few-Shot tasks
* `zero.py`: Zero-Shot tasks
* `run.py`: Main experiment
* `requirements.txt`: all requires for training

### Main Experiments

1. Install requirements. `pip install -r requirements.txt`

2. Download LLM Models. You can download all llms on  `Huggingface`, and then save them in `/llm/gpt2, /llm/Llama-3.2-1B, /llm/Qwen2.5-0.5B`

3. Training. All the scripts are in the directory `./scripts`. For example, if you want to get training for gpt2 on Actions-im dataset-1,  just run the following command, and you can open `./logs/gpt2/action-im` to see the results once the training is done:

   ```bash
   bash ./scripts/action-im.sh
   ```

### Few-Shot and Zero-Shot Experiments

There are some examples in `./scripts`. For example, if you want to get training for gpt2 on Phonation dataset-1 on Zero-Shot,  just run the following command, and you can open `./logs/zero/pro1_zero_gpt2.log` to see the results once the training is done:

```bash
bash ./scripts/pro1_zero.sh
```

