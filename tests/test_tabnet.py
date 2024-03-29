import torch
from ..src.layers import FastGLU


def test_GLU_forward():
    tensor = torch.Tensor([[1,2,3,4]])
    output = FastGLU(4).forward(tensor)
    assert tensor.size(dim = 1) == output.size(dim =1 )
