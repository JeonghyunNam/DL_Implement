import torch
import torch.nn as nn

if __name__ == '__main__':
    t = torch.randn(2,3,1)
    print(f"t is : {t}")
    print(f"size of t is {t.size()}")
    print(f"unsqueeze 1 : {t.unsqueeze(1)}")
    print(f"size of unsqueeze 1 {t.unsqueeze(1).size()}")
    print(f"unsqueeze 0 : {t.unsqueeze(0)}")