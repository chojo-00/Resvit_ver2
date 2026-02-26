import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import ms_ssim

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(16)])  # relu3_4
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(16, 25)])  # relu4_4
        
        for param in self.parameters():
            param.requires_grad = False

        self.eval()


    def forward(self, x, y):
        # Step 1: [-1, 1] → [0, 1] 변환
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
        
        # Step 2: 1채널 → 3채널
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # Step 3: ImageNet normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Step 4: Feature 추출
        x_feat1 = self.slice1(x)
        y_feat1 = self.slice1(y)
        x_feat2 = self.slice2(x_feat1)
        y_feat2 = self.slice2(y_feat1)
        
        loss = F.l1_loss(x_feat1, y_feat1) + F.l1_loss(x_feat2, y_feat2)
        return loss



class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()
    
    def forward(self, x, y):
        # ms_ssim은 [0,1] 범위 가정
        # 모델 출력은 [-1,1]이므로 [0,1]로 변환
        x = (x + 1) / 2.0
        y = (y + 1) / 2.0
        return 1 - ms_ssim(x, y, data_range=1.0, size_average=True)