import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_h=32, nc=1, nclass=37, nh=256):
        super(CRNN, self).__init__()
        assert img_h % 16 == 0
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, batch_first=False),
        )
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        seq, _ = self.rnn[0](conv)
        seq = self.fc(seq)
        return seq
