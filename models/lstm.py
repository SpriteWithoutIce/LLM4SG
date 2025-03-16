import torch.nn as nn

class LSTM4TS(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0", lstm_layers=2, hidden_size=256):
        super(LSTM4TS, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=lstm_layers, bidirectional=True, batch_first=True)

        # Classifier
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_output, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        cls_embedding = lstm_output[:, 0, :]  # (batch_size, hidden_size * 2)
        logits = self.classifier(cls_embedding)  # (batch_size, num_classes)
        return logits