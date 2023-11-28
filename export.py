import torch
from networks.wide_resnet import Wide_ResNet


if __name__ == "__main__":
    model = Wide_ResNet(28, 10, 0.3, 10)
    x = torch.randn(1, 3, 32, 32)
    try:
        torch.export.export(model, (x, ))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e
