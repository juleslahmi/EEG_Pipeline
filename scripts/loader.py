from braindecode import EEGClassifier
import torch
import numpy as np

def load_skorch_model(model, f_params, device="cuda", classes=None):
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    clf = EEGClassifier(
        module=model,
        device=("cuda" if use_cuda else "cpu"),
        train_split=None,
        classes=(np.array(classes) if classes is not None else None),
    )
    clf.initialize()
    clf.load_params(f_params=f_params)
    return clf

if __name__ == "__main__":
    # Example usage:
    # model = build_model("tcn", n_chans=32, n_classes=2, n_times=512)
    # clf = load_skorch_model(model, "best_params.pt", device="cuda", classes=[0,1])
    print("This is a minimal loading helper. See README.md.")