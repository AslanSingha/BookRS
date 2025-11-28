import numpy as np
from backend.core.config import ART_DIR
import os

uf = np.load(os.path.join(ART_DIR, "als_user_factors.npz"))["data"]
it = np.load(os.path.join(ART_DIR, "als_item_factors.npz"))["data"]
print("User factors shape:", uf.shape)
print("Item factors shape:", it.shape)
