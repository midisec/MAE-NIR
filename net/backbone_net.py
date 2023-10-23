import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2).float() * (-math.log(10000.0) / input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class AutoEncoderWithPositional(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoderWithPositional, self).__init__()

        self.position_enc = PositionalEncoding(input_size)

        # Calculating the sizes of the layers dynamically based on the input size
        layer_sizes = [input_size]
        while layer_sizes[-1] > 256:
            layer_sizes.append(layer_sizes[-1] * 3 // 4)

        # Encoder
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        self.Encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        for i in reversed(range(len(layer_sizes) - 1)):
            layers.append(nn.Linear(layer_sizes[i + 1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers[-1] = nn.Tanh()
        self.Decoder = nn.Sequential(*layers)

    def forward(self, input):
        code = input.view(input.size(0), -1)
        code = self.position_enc(code)
        code = self.Encoder(code)
        output = self.Decoder(code)
        output = output.view(input.size(0), 1, input.size(1))
        return output


class MaskedAutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(MaskedAutoEncoder, self).__init__()

        # Calculating the sizes of the layers dynamically based on the input size
        layer_sizes = [input_size]
        while layer_sizes[-1] > 256:
            layer_sizes.append(layer_sizes[-1] * 3 // 4)

        # Encoder
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        self.Encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        for i in reversed(range(len(layer_sizes) - 1)):
            layers.append(nn.Linear(layer_sizes[i+1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers[-1] = nn.Tanh()
        self.Decoder = nn.Sequential(*layers)

    def random_masked(self, masking_rate=0.5, input_tensor=None):
        if input_tensor is None:
            raise ValueError("Input tensor should not be None.")
        mask = torch.bernoulli(masking_rate * torch.ones(input_tensor.size())).to(input_tensor.device)
        return input_tensor * mask

    def forward(self, input):
        code = input.view(input.size(0), -1)
        code = self.random_masked(masking_rate=0.5, input_tensor=code)
        code = self.Encoder(code)
        output = self.Decoder(code)
        output = output.view(input.size(0), 1, input.size(1))
        return output


class MaskedAutoEncoder1DStageI(nn.Module):
    def __init__(self):
        super(MaskedAutoEncoder1DStageI, self).__init__()
        self.masking_rate = 0.5
        # self.Encoder = nn.Sequential(
        #     nn.Linear(4199, 3096),
        #     nn.ReLU(),
        #     nn.Linear(3096, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        # )
        #
        # self.Decoder = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 3096),
        #     nn.ReLU(),
        #     nn.Linear(3096, 4199),
        #     # nn.Sigmoid()
        #     nn.Tanh()
        # )
        self.Encoder = nn.Sequential(
            nn.Linear(4199, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4199),
            # nn.Sigmoid()
            nn.Tanh()
        )


    def forward(self, input):
        code = input.view(input.size(0), -1)
        for i in range(code.shape[0]):
            mask = torch.bernoulli(self.masking_rate * torch.ones(code[i].size())).cuda()
            code[i] = code[i] * mask
        code = self.Encoder(code)
        output = self.Decoder(code)
        output = output.view(input.size(0), 4199)
        return output


class MaskedAutoEncoder1DStageII(nn.Module):
    def __init__(self):
        super(MaskedAutoEncoder1DStageII, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(227, 200),
            nn.ReLU(),
            nn.Linear(200, 196),
            nn.ReLU(),
            nn.Linear(196, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 196),
            nn.ReLU(),
            nn.Linear(196, 200),
            nn.ReLU(),
            nn.Linear(200, 227),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, input):
        code = input.view(input.size(0), -1)
        code = self.Encoder(code)
        output = self.Decoder(code)
        output = output.view(input.size(0), 227)
        return output


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.layer(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()
        self.encoder1 = UNetDown(in_channels, 64)
        self.encoder2 = UNetDown(64, 128)
        self.encoder3 = UNetDown(128, 256)
        self.encoder4 = UNetDown(256, 512)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder4 = UNetUp(1024, 512)
        self.decoder3 = UNetUp(512, 256)
        self.decoder2 = UNetUp(256, 128)
        self.decoder1 = UNetUp(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        center = self.center(x4)
        x = self.decoder4(center, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        return self.final(x)


class MAE2DStageI(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MAE2DStageI, self).__init__()
        # Define the encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Define the decoder part
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MAE2DStageII(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MAE2DStageII, self).__init__()
        # This can be more complex as per your requirements
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    mae = MaskedAutoEncoder(228)
    print(mae)