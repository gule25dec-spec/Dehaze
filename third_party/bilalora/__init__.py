"""BiLaLoRA (CVPR 2026) inference wrapper.

Base network (`Backbonecs`, a re-parameterised DEA-Net-style backbone) + LoRA layers injected into
the sub-modules chosen by the bilevel layer search, exactly as in the authors' BiLaLoRA_Test.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DEAmodel import Backbonecs


class LoRALayer(nn.Module):
    """Copied from BiLaLoRA_Test.py (official)."""
    def __init__(self, original_layer: nn.Conv2d, r: int, lora_alpha: int, shared_alpha: nn.Parameter = None):
        super().__init__()
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False
        in_ch, out_ch, k = original_layer.in_channels, original_layer.out_channels, original_layer.kernel_size[0]
        self.r, self.lora_alpha, self.scaling = r, lora_alpha, lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r * k, in_ch * k))
        self.lora_B = nn.Parameter(torch.zeros(out_ch * k, r * k))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.alpha = shared_alpha if shared_alpha is not None else nn.Parameter(torch.zeros(1))

    def forward(self, x):
        lora_weight = (self.lora_B @ self.lora_A).view(self.original_layer.weight.shape)
        w = self.original_layer.weight + torch.sigmoid(self.alpha) * self.scaling * lora_weight
        o = self.original_layer
        return F.conv2d(x, w, o.bias, o.stride, o.padding, o.dilation, o.groups)


def inject_lora_to_layers(model, target_layers, r, lora_alpha):
    """Copied from BiLaLoRA_Test.py (official): one shared gate per selected sub-module."""
    for layer_name in target_layers:
        parent = model.get_submodule(layer_name)
        shared_alpha = nn.Parameter(torch.zeros(1))
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(parent, child_name, LoRALayer(child, r, lora_alpha, shared_alpha))
    return model


def build_bilalora(base_path, lora_path=None):
    net = Backbonecs()
    sd = torch.load(base_path, map_location="cpu", weights_only=False)
    net.load_state_dict(sd)
    if lora_path is not None:
        ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
        cfg = ckpt["nas_lora_config"]
        net = inject_lora_to_layers(net, cfg["selected_layers"], cfg["r"], cfg["lora_alpha"])
        missing, unexpected = net.load_state_dict(ckpt["lora_state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected LoRA keys: {unexpected[:5]}")
        net.lora_layers = cfg["selected_layers"]
    return net
