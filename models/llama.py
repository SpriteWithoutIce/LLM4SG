import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class Llama3TS(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0", llama_layers=6, lora=False, lstm=True):
        super(Llama3TS, self).__init__()
        print(lora, lstm)
        self.device = device
        # Load Llama3 Model
        self.tokenizer = AutoTokenizer.from_pretrained("./llm/Llama-3.2-1B")
        self.llama3 = AutoModelForCausalLM.from_pretrained("./llm/Llama-3.2-1B", output_hidden_states=True)
        # print("Llama3 Loaded")
        print(self.llama3)

        llama_layers = min(llama_layers, len(self.llama3.model.layers))

        self.llama3.model.layers = self.llama3.model.layers[:llama_layers]
        print(f"Llama3 Loaded with {llama_layers} layers")

        # 定义 LoRA 配置
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,  # 缩放因子
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
            lora_dropout=0.1,
            bias="none",
        )
        if lora==True:
            self.llama3 = get_peft_model(self.llama3, lora_config)
            
        for name, param in self.llama3.named_parameters():
            if "norm" in name or "rotary_emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Define embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Conv1d(in_channels=64, out_channels=self.llama3.config.hidden_size,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.llama3.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2), 
        )
        # lstm
        self.lstm = nn.LSTM(input_size=self.llama3.config.hidden_size, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True)
        # Classifier to predict the output classes
        if lstm==True:
            self.classifier = nn.Sequential(
                nn.Linear(512*2, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.llama3.config.hidden_size, num_classes)


    def forward(self, x, lstm=True):
        x = x.permute(0, 2, 1)
        x = self.embedding_layer(x)
        x = x.permute(0, 2, 1)
        
        outputs = self.llama3(inputs_embeds=x)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        if lstm==True:
            # lstm
            lstm_output, _ = self.lstm(last_hidden_state)
            cls_embedding = lstm_output[:, 0, :]
        else:
            cls_embedding = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)  # (batch_size, num_classes)
        return logits
