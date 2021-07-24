import torchvision.models as models
import numpy as np
import torch


class WarmUpGPUs():
    def __init__(self, device):
        self.device = device
        self.resnet152 = models.resnet152(pretrained=True)

    def simple_warmup(self, warmup_iteration):
        print(f"Simple Warmup Model: ResNet-152/pretrained")
        print(f"\nWarmup Start(iteration: {warmup_iteration})")

        dummy_input = torch.randn(
            1, 3,224,224,dtype=torch.float).to(self.device)
        self.resnet152 = self.resnet152.to(self.device)

        for _ in range(warmup_iteration):
            _ = self.resnet152(dummy_input)

        print("Warmup Finish!\n")
        # print("*"*40)
        

        
        