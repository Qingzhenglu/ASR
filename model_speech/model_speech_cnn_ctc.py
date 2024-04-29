import torch.nn as nn
import torch


class DFCNN(nn.Module):
    def __init__(self, vocab_size, is_train=True):
        super(DFCNN, self).__init__()
        self.vocab_size = vocab_size
        self.is_training = is_train
        self._model_init()

    def _model_init(self):
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(num_features=32, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=32, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=0.0002),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(1),
        )
        self.layer5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=0.0002),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(1),
        )

        self.drop = nn.Dropout(0.2)
        self.dense = nn.Linear(3200, 256)
        self.out = nn.Linear(256, out_features=self.vocab_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1, 3200)
        x = self.drop(x)
        x = self.dense(x)
        x = self.drop(x)
        x = self.out(x)
        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# m = DFCNN(1176).to(device)
# summary(m, (1, 1600, 200), 20)
