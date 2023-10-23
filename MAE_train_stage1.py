import os
import torch
from torch import nn, optim
from data.load_dataset import SpectralDataSet
from torch.utils.data import DataLoader
from net.backbone_net import MaskedAutoEncoder1DStageI
from net.losses import DBSNLoss, DBSNLoss_Pretrain, MAPLoss, MAPLoss_Pretrain, L1Loss

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 0.00003
epochs = 500


Lucas_spectral_data = SpectralDataSet("LucasSoil")
# Lucas_spectral_data = SpectralDataSet("LucasSoil", preprocessing="MSC")
Lucas_spectral_data_loader = DataLoader(Lucas_spectral_data, shuffle=True, batch_size=32, num_workers=0)
sign = 'LUCAS'
path_prefix = "./save_model/%s" % sign

if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)

MAE1D = MaskedAutoEncoder1DStageI()
MAE1D.to(device)

# criteon = L1Loss().cuda()
criteon = nn.MSELoss()
optimizer = optim.Adam(MAE1D.parameters(), lr=learning_rate)

for epoch in range(epochs):

    for idx, (spectrum, _) in enumerate(Lucas_spectral_data_loader):
        spectrum = spectrum.float().to(device)
        reconstructed_spectrum = MAE1D(spectrum)
        loss = criteon(reconstructed_spectrum, spectrum.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 25 == 0:
            print(
                "epoch:{}/{}, batchidx:{}/{}, Loss:{}".format(epoch, epochs, idx, len(Lucas_spectral_data_loader), loss))

    if epoch % 100 == 1:
        path1 = "%s/mae1d_%s.ckpt" % (path_prefix, epoch)
        path2 = "%s/mae1d_%s.pt" % (path_prefix, epoch)
        torch.save(MAE1D.state_dict(), path1)
        torch.save(MAE1D, path2)
        print("save model done. path=%s" % path2)

