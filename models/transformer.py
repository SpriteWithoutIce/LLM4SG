import torch
import torch.nn as nn
import math

class VanillaTransformer(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0"):
        super(VanillaTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.device = device

        # 位置编码
        self.positional_encoding = self._generate_positional_encoding(seq_len, input_dim).to(device)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,  # 注意力头数
            dim_feedforward=256,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=1,  # 注意力头数
            dim_feedforward=256,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # 分类器
        self.classifier = nn.Linear(input_dim * seq_len, num_classes)

    def _generate_positional_encoding(self, seq_len, d_model):
        """生成位置编码"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def forward(self, x):
        # 添加位置编码
        x = x + self.positional_encoding

        # 调整输入形状以适应 Transformer
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)

        # 通过 Transformer Encoder
        memory = self.transformer_encoder(x)

        # 通过 Transformer Decoder
        # 这里使用 memory 作为 decoder 的输入
        output = self.transformer_decoder(x, memory)

        # 调整形状以进行分类
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, input_dim)
        output = output.reshape(output.size(0), -1)  # (batch_size, seq_len * input_dim)

        # 分类
        output = self.classifier(output)
        return output