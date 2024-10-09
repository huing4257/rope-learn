import torch
from torch import Tensor

def rotary_pos_encoding(
    x: Tensor
) -> Tensor:
    torch.ops.rope.rotary_pos_encoding(x)