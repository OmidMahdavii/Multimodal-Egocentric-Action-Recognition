from torch import nn


class EMG_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=16, hidden_size=5, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=5, hidden_size=50, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(50, num_classes),
            nn.Softmax()
        )

        

    def forward(self, x):
        y, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        y = self.fc(y[:, -1, :])  # Take the output of the last time step
        return y, {}


class EMG_CNN(nn.Module):
    def __init__(self, num_classes, num_clips):
        super().__init__()

        self.model = nn.Sequential(
            # Input shape: (num_clips, 1024)
            nn.Conv1d(in_channels=num_clips, out_channels=num_clips*2, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(num_clips*2),
            nn.ReLU(),
            # Shape: (num_clips*2, 256)
            nn.Conv1d(in_channels=num_clips*2, out_channels=num_clips*4, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(num_clips*4),
            nn.ReLU(),
            # Shape: (num_clips*4, 64)
            nn.Conv1d(in_channels=num_clips*4, out_channels=num_clips*8, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(num_clips*8),
            nn.ReLU(),
            # Shape: (num_clips*8, 16)
            nn.Conv1d(in_channels=num_clips*8, out_channels=num_classes, kernel_size=16, stride=1, padding=0),
            # Shape: (num_classes, 1)
            nn.Sigmoid()
        )

        

    def forward(self, x):
        # x = x.unsqueeze(1)
        # y = self.model(x)
        # return y.squeeze(), {}
        return self.model(x).squeeze(), {}
    
