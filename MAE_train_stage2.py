import os
import torch
from torch import nn, optim
from data.load_dataset import SpectralDataSet
from torch.utils.data import DataLoader
from net.backbone_net import MaskedAutoEncoder1DStageI, MaskedAutoEncoder1DStageII
from net.losses import DBSNLoss, DBSNLoss_Pretrain, MAPLoss, MAPLoss_Pretrain, L1Loss
from copy import deepcopy

StageIModelPath1 = "./save_model/LUCAS/mae1d_1.ckpt"
StageIModelPath2 = "./save_model/LUCAS/mae1d_1.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 0.000003
epochs = 500


AnhuiSoil_spectral_data = SpectralDataSet("AnhuiSoil")
# AnhuiSoil_spectral_data = SpectralDataSet("AnhuiSoil", preprocessing="MSC")
AnhuiSoil_spectral_data_loader = DataLoader(AnhuiSoil_spectral_data, shuffle=True, batch_size=32, num_workers=0)
sign = 'AnhuiSoil'
path_prefix = "./save_model/%s" % sign

if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)


MAE1D_StageI = torch.load(StageIModelPath2)
MAE1D_StageI.load_state_dict(torch.load(StageIModelPath1))


def para_state_dict(model, model_save_path):
    # model: new model
    # model_save_dir : AE.ckpt path

    state_dict = deepcopy(model.state_dict())
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        for key in state_dict:  # 在新的网络模型中遍历对应参数
            if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
                print("成功初始化参数:", key)
                state_dict[key] = loaded_paras[key]
    return state_dict


print("step1: ")
for param_tensor in MAE1D_StageI.state_dict():
    print(param_tensor, "\t", MAE1D_StageI.state_dict()[param_tensor].size())

print("step2")
MAE1D_StageII = MaskedAutoEncoder1DStageII()
for param_tensor in MAE1D_StageII.state_dict():
    print(param_tensor, "\t", MAE1D_StageII.state_dict()[param_tensor].size())

print("init...")
state_dict = para_state_dict(MAE1D_StageII, StageIModelPath1)
MAE1D_StageII.load_state_dict(state_dict)


print("frozen....")
# need_frozen_list = ["Encoder.0.bias","Encoder.2.weight","Encoder.2.bias","Encoder.4.weight","Encoder.4.bias","Encoder.6.weight","Encoder.6.bias","Encoder.8.weight","Encoder.8.bias","Decoder.0.weight","Decoder.0.bias","Decoder.2.weight","Decoder.2.bias","Decoder.4.weight","Decoder.4.bias","Decoder.6.weight","Decoder.6.bias"]
need_frozen_list = ["Encoder.8.weight","Encoder.8.bias","Decoder.0.weight","Decoder.0.bias"]
for param in MAE1D_StageII.named_parameters():
    if param[0] in need_frozen_list:
        param[1].requires_grad = False
        print(param[0] + "has frozen")

MAE1D_StageII.to(device)
# criteon = L1Loss().cuda()
criteon = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, MAE1D_StageII.parameters()), lr=learning_rate)

for epoch in range(epochs):

    for idx, (spectrum, _) in enumerate(AnhuiSoil_spectral_data_loader):
        spectrum = spectrum.float().to(device)
        reconstructed_spectrum = MAE1D_StageII(spectrum)
        loss = criteon(reconstructed_spectrum, spectrum.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 25 == 0:
            print(
                "epoch:{}/{}, batchidx:{}/{}, Loss:{}".format(epoch, epochs, idx, len(AnhuiSoil_spectral_data_loader), loss))

    if epoch % 100 == 1:
        path1 = "%s/mae1d_%s.ckpt" % (path_prefix, epoch)
        path2 = "%s/mae1d_%s.pt" % (path_prefix, epoch)
        torch.save(MAE1D_StageII.state_dict(), path1)
        torch.save(MAE1D_StageII, path2)
        print("save model done. path=%s" % path2)
