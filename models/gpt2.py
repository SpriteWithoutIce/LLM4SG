from transformers import GPT2Model
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class GPT4TS(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0", gpt_layers=6, lora=True, lstm=True):
        super(GPT4TS, self).__init__()
        print(lora, lstm)
        # GPT2 Model
        self.gpt2 = GPT2Model.from_pretrained(
            "./llm/gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        print("GPT2 Loaded with {} layers".format(gpt_layers))

        # 定义 LoRA 配置
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,
            target_modules=["attn.c_attn", "attn.c_proj"],  # 目标模块
            lora_dropout=0.1,
            bias="none",
        )

        if lora == True:
            self.gpt2 = get_peft_model(self.gpt2, lora_config)
        for name, param in self.gpt2.named_parameters():
            # param.requires_grad = True
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.pos_encoder = PositionalEncoding(self.gpt2.config.n_embd)
        self.embedding_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=256, out_channels=self.gpt2.config.n_embd,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=self.gpt2.config.n_embd, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        if lstm==True:
            self.classifier = nn.Linear(256*2, num_classes)
        else:
            self.classifier = nn.Linear(self.gpt2.config.n_embd, num_classes)

    def forward(self, x, lstm=True):
        x = x.permute(0, 2, 1)  
        x = self.embedding_layer(x)  
        x = x.permute(0, 2, 1)  
        gpt2_output = self.gpt2(inputs_embeds=x).last_hidden_state
        if lstm==True:
            lstm_output, _ = self.lstm(gpt2_output)
            cls_embedding = lstm_output[:, 0, :]
        else:
            cls_embedding = gpt2_output[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
