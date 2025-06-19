import os
import sys
import pickle
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect pickle files for DAMSM')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset name: coco, bird')
    args = parser.parse_args()
    return args

def load_pickle_file(file_path):
    """Load a pickle file and return its contents"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def inspect_captions_pickle(file_path):
    """Inspect the content of captions_DAMSM.pickle file"""
    print(f"\n{'='*20} Inspecting captions_DAMSM.pickle {'='*20}")
    print(f"File path: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return
    
    data = load_pickle_file(file_path)
    if data is None:
        return
    
    print(f"Data type: {type(data)}")
    
    if isinstance(data, tuple):
        print(f"Tuple length: {len(data)}")
        for i, item in enumerate(data):
            try:
                print(f"\nTuple item {i}:")
                print(f"  Type: {type(item)}")
                
                if isinstance(item, dict):
                    print(f"  Dictionary size: {len(item)}")
                    print(f"  Keys: {list(item.keys())[:5]}..." if len(item) > 5 else f"  Keys: {list(item.keys())}")
                    
                    # Show a sample entry
                    if len(item) > 0:
                        sample_key = next(iter(item))
                        print(f"\n  Sample entry for key '{sample_key}':")
                        print(f"    {item[sample_key]}")
                elif isinstance(item, list):
                    print(f"  List length: {len(item)}")
                    print(f"  Sample items (first 5): {item[:5]}")
                elif isinstance(item, np.ndarray):
                    print(f"  Array shape: {item.shape}")
                    print(f"  Array dtype: {item.dtype}")
                    print(f"  Sample data: {item[:5]}")
                else:
                    print(f"  Value: {str(item)[:100]}..." if len(str(item)) > 100 else f"  Value: {item}")
            except Exception as e:
                print(f"  Error inspecting item {i}: {e}")
    else:
        print(f"Content: {data}")

def inspect_filenames_pickle(file_path):
    """Inspect the content of filenames.pickle file"""
    print(f"\n{'='*20} Inspecting filenames.pickle {'='*20}")
    print(f"File path: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return
    
    data = load_pickle_file(file_path)
    if data is None:
        return
    
    print(f"Data type: {type(data)}")
    
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        print("Sample filenames (first 10):")
        for i in range(min(10, len(data))):
            print(f"  {data[i]}")
    elif isinstance(data, dict):
        print(f"Dictionary size: {len(data)}")
        print(f"Keys: {list(data.keys())}")
        
        # Show sample values
        for key in list(data.keys())[:2]:
            print(f"\nSample for '{key}':")
            if isinstance(data[key], list) and len(data[key]) > 0:
                print(f"  List length: {len(data[key])}")
                print(f"  First 5 items: {data[key][:5]}")
            else:
                print(f"  {data[key]}")
    else:
        print(f"Content type: {type(data)}")
        print(f"Content preview: {str(data)[:100]}..." if len(str(data)) > 100 else f"Content: {data}")

def main(args):
    dataset = args.dataset.lower()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Set paths based on dataset
    if dataset == 'coco':
        data_dir = os.path.join(base_dir, "data", "coco")
    elif dataset == 'bird':
        data_dir = os.path.join(base_dir, "data", "birds")
    else:
        print(f"Unknown dataset: {dataset}")
        return
    
    # Inspect captions_DAMSM.pickle
    captions_file = os.path.join(data_dir, "captions_DAMSM.pickle")
    inspect_captions_pickle(captions_file)
    
    # Check for filenames.pickle in different possible locations
    possible_filename_paths = [
        os.path.join(data_dir, "filenames.pickle"),
        os.path.join(data_dir, "train", "filenames.pickle"),
        os.path.join(data_dir, "val", "filenames.pickle"),
        os.path.join(data_dir, "test", "filenames.pickle")
    ]
    
    for path in possible_filename_paths:
        if os.path.exists(path):
            inspect_filenames_pickle(path)
    
    print("\nDone!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
