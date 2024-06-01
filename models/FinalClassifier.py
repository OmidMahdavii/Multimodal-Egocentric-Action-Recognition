from torch import nn


class Classifier(nn.Module):
    def __init__(self, num_classes, num_clips):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

        self.cnn = nn.Sequential(
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
            # Shape: (num_clips*4, 16)
            nn.Conv1d(in_channels=num_clips*8, out_channels=num_classes, kernel_size=16, stride=1, padding=0),
            # Shape: (num_classes, 1)
            nn.Sigmoid()
        )

        

    def forward(self, x):
        return self.cnn(x), {}
