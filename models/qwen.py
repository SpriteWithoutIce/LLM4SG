import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class QwenTS(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0", qwen_layers=6, lora=False, lstm=True):
        super(QwenTS, self).__init__()
        print(lora, lstm)
        # Load Qwen2.5-0.5B Model
        self.tokenizer = AutoTokenizer.from_pretrained("./llm/Qwen2.5-0.5B")
        self.qwen = AutoModelForCausalLM.from_pretrained("./llm/Qwen2.5-0.5B", output_hidden_states=True)
        print(self.qwen)
        # 限制 Qwen 层数
        qwen_layers = min(qwen_layers, len(self.qwen.model.layers))
        self.qwen.model.layers = self.qwen.model.layers[:qwen_layers]
        print(f"Qwen Loaded with {qwen_layers} layers")

        # 定义 LoRA 配置
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,  # 缩放因子
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
            lora_dropout=0.1,
            bias="none",
        )

        if lora == True:
            self.qwen = get_peft_model(self.qwen, lora_config)
        
        for name, param in self.qwen.named_parameters():
            if "ln" in name or "rotary_pos_emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 获取 hidden size
        hidden_size = self.qwen.config.hidden_size

        # 定义输入数据的嵌入层
        self.embedding_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            
            nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2), 
        )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        
        # 分类层
        if lstm == True:
            self.classifier = nn.Linear(256*2, num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lstm=True):
        x = x.permute(0, 2, 1)
        x = self.embedding_layer(x)
        x = x.permute(0, 2, 1)
        
        outputs = self.qwen(inputs_embeds=x)
        hidden_states = outputs.hidden_states

        last_hidden_state = hidden_states[-1]
        if lstm==True:
            lstm_output, _ = self.lstm(last_hidden_state) 
            cls_embedding = lstm_output[:, 0, :] 
        else:
            cls_embedding = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)  # (batch_size, num_classes)
        return logits
