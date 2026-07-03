import torch
import numpy as np
import os, sys

from model import SmallTransformer
from utils import DEVICE, P, generate_all_pairs

ARTIFACTS = "artifacts"
BATCH_SIZE = 256

OP = sys.argv[1] if len(sys.argv) > 1 else "add"
SUFFIX = "" if OP == "add" else "_mult"
MODEL_DIR = f"{ARTIFACTS}/small{SUFFIX}"
ACT_PATH = f"{ARTIFACTS}/small_model_activations{SUFFIX}.npy"
LBL_PATH = f"{ARTIFACTS}/mod_arithmetic_labels{SUFFIX}.npy"


def main():
    if os.path.exists(ACT_PATH) and os.path.exists(LBL_PATH):
        print(f"[cache] {OP} activations already cached, loading...")
        acts = np.load(ACT_PATH)
        lbls = np.load(LBL_PATH)
        print(f"  {acts.shape[0]} activations [{acts.shape[1]}], {lbls.shape[0]} labels")
        return

    print(f"[cache] Running small model ({OP} mod {P}) on all P^2 pairs...")
    model = SmallTransformer().to(DEVICE)
    state = torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_inputs, all_labels = generate_all_pairs(op=OP)
    all_acts = []
    for i in range(0, len(all_inputs), BATCH_SIZE):
        x = all_inputs[i:i+BATCH_SIZE]
        with torch.no_grad():
            _, acts = model(x, return_activations=True)
        batch_acts = acts["blocks.1.hook_resid_post"][:, 1, :].numpy()
        all_acts.append(batch_acts)

    acts_arr = np.concatenate(all_acts, axis=0)
    lbls_arr = all_labels.numpy()

    np.save(ACT_PATH, acts_arr)
    np.save(LBL_PATH, lbls_arr)
    print(f"[cache] Saved {acts_arr.shape[0]} activations [{acts_arr.shape[1]}] + labels")


if __name__ == "__main__":
    main()
