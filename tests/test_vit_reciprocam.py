import pytest
from src.vit_recipro_cam import VitReciproCam
import torch
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights

def test_vit_recipro_cam():
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to('cpu')
    model.eval()
    model = model.cpu()
    vit_reciprocam = VitReciproCam(model, is_gaussian=True, cls_token=True)
    input_tensor = torch.randn(1, 3, 224, 224).cpu()
    cam, class_id = vit_reciprocam(input_tensor.unsqueeze(0))
    assert cam.shape == (1, 3, 224, 224)