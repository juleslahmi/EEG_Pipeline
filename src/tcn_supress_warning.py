import torch.nn as nn

def replace_dropout2d(m: nn.Module):
    for name, child in list(m.named_children()):
        if isinstance(child, nn.Dropout2d):
            new_layer = nn.Identity() if child.p == 0 else nn.Dropout1d(p=child.p)
            setattr(m, name, new_layer)
        else:
            replace_dropout2d(child)