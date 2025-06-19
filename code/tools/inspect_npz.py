import numpy as np
import sys
import os

def inspect_npz(npz_path):
    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        return
    data = np.load(npz_path)
    print(f"Keys in {npz_path}: {list(data.keys())}")
    for key in data:
        arr = data[key]
        print(f"Key: {key}, dtype: {arr.dtype}, shape: {arr.shape}")
        print(f"Sample values for {key}: {arr.flatten()[:5]}")
    data.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <path_to_npz>")
    else:
        inspect_npz(sys.argv[1])
