from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGNet, BDTCN, HybridNet, BIOT

def build_model(model_cfg):
    """
    Parameters
    ----------
    model_cfg : dict
        Expected keys:
          - name: 'shallow' | 'deep4' | 'eegnet'
          - n_chans: int
          - n_times: int
          - n_classes: int
        Optional per-model kwargs (e.g., drop_prob).
    Returns
    -------
    torch.nn.Module
        Initialized model (CPU or moved to CUDA by train.py).
    """
    name      = model_cfg["name"].lower()
    n_chans   = int(model_cfg["n_chans"])
    n_times   = int(model_cfg["n_times"])
    n_classes = int(model_cfg["n_classes"])
    
    if name == 'shallow':
        model = ShallowFBCSPNet(
            n_chans,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto"
        ).cuda()

    elif name == 'deep4':
        model = Deep4Net(
            n_chans,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto"
        ).cuda()

    elif name == 'eegnet':
        model = EEGNet(
            n_chans,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto"
        ).cuda()

    elif name == 'tcn':
        model = BDTCN(
            n_chans,
            n_outputs=n_classes,
            drop_prob=0,
            n_blocks=1,        # default is 4
            kernel_size=2,     # default is 40
        ).cuda()

    elif name == 'hybridnet':
        model = HybridNet(
            n_chans,
            n_outputs=n_classes,
            n_times=n_times,
        ).cuda()

    elif name == 'biot':
        model = BIOT(
                n_chans=n_chans,
                n_outputs=n_classes,
                n_times=n_times, 
        ).cuda()

    else: 
        raise ValueError(f"Unknown model name: {name}")
    
    return model
