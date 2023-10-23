import os
import torch
from torch import nn, optim
from data.load_dataset import SpectralDataSet
from torch.utils.data import DataLoader
from net.backbone_net import MaskedAutoEncoder1DStageI
from net.losses import DBSNLoss, DBSNLoss_Pretrain, MAPLoss, MAPLoss_Pretrain, L1Loss

device = "cuda" if torch.cuda.is_available() else "cpu"


AnhuiSoil_spectral_data = SpectralDataSet("AnhuiSoil")
# AnhuiSoil_spectral_data = SpectralDataSet("AnhuiSoil", preprocessing="MSC")
AnhuiSoil_spectral_data_loader = DataLoader(AnhuiSoil_spectral_data, shuffle=True, batch_size=1, num_workers=0)


StageIIModelPath1 = "./save_model/AnhuiSoil/mae1d_401.ckpt"
StageIIModelPath2 = "./save_model/AnhuiSoil/mae1d_401.pt"

MAE1D_StageII = torch.load(StageIIModelPath2)
MAE1D_StageII.load_state_dict(torch.load(StageIIModelPath1))
MAE1D_StageII.to(device)

spectrum, _ = iter(AnhuiSoil_spectral_data_loader).__next__()
spectrum = spectrum.float().to(device)
reconstructed_spectrum = MAE1D_StageII(spectrum.float())

print(fakeseqs)





