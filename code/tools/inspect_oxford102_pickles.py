import os
import pickle
import numpy as np
from collections import defaultdict

def inspect_filenames_pickle(filepath, limit=10):
    """Inspect a filenames pickle file and display statistics and samples."""
    print(f"\n==================== Inspecting {os.path.basename(filepath)} ====================")
    print(f"File path: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"List length: {len(data)}")
            
            if len(data) > 0:
                print(f"Sample filenames (first {min(limit, len(data))}):")
                for i, item in enumerate(data[:limit]):
                    print(f"  {item}")
                
                if len(data) > 2*limit:
                    print("  ...")
                    
                    print(f"Sample filenames (last {min(limit, len(data))}):")
                    for i, item in enumerate(data[-limit:]):
                        print(f"  {item}")
            
            # Analyze filename patterns
            prefix_counts = defaultdict(int)
            for item in data:
                if isinstance(item, str) and '_' in item:
                    prefix = item.split('_')[0]
                    prefix_counts[prefix] += 1
            
            if prefix_counts:
                print("\nFilename prefix distribution:")
                for prefix, count in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {prefix}: {count} files ({count/len(data)*100:.2f}%)")
        else:
            print("Not a list - unexpected format")
    
    except Exception as e:
        print(f"Error reading file: {e}")

def inspect_captions_pickle(filepath, limit=3):
    """Inspect a captions pickle file and display statistics and samples."""
    print(f"\n==================== Inspecting {os.path.basename(filepath)} ====================")
    print(f"File path: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"List length: {len(data)}")
            
            # Statistics about captions
            caption_lengths = []
            captions_per_image = []
            
            for i, item in enumerate(data):
                if isinstance(item, list):
                    captions_per_image.append(len(item))
                    
                    for cap in item:
                        if isinstance(cap, list):
                            caption_lengths.append(len(cap))
            
            if caption_lengths:
                print(f"\nCaption statistics:")
                print(f"  Average caption length: {np.mean(caption_lengths):.2f} tokens")
                print(f"  Min caption length: {min(caption_lengths)} tokens")
                print(f"  Max caption length: {max(caption_lengths)} tokens")
            
            if captions_per_image:
                unique_counts = set(captions_per_image)
                print(f"  Captions per image: {', '.join(map(str, sorted(unique_counts)))}")
            
            # Display sample captions
            if len(data) > 0 and isinstance(data[0], list) and len(data[0]) > 0:
                print(f"\nSample captions from first {min(limit, len(data))} images:")
                for i, item in enumerate(data[:limit]):
                    print(f"  Image {i}:")
                    for j, cap in enumerate(item[:3]):  # Show first 3 captions
                        print(f"    Caption {j}: {cap}")
        else:
            print("Not a list - unexpected format")
    
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    """Main function to inspect all pickle files."""
    base_dir = os.path.join('c:', os.sep, 'Users', 'nanda', 'OneDrive', 'Desktop', 'DF-GAN')
    oxford_dir = os.path.join(base_dir, 'data', 'flower')
    
    # Inspect train filenames
    train_path = os.path.join(oxford_dir, 'train', 'filenames.pickle')
    if os.path.exists(train_path):
        inspect_filenames_pickle(train_path)
    else:
        print(f"File not found: {train_path}")
    
    # Inspect test filenames
    test_path = os.path.join(oxford_dir, 'test', 'filenames.pickle')
    if os.path.exists(test_path):
        inspect_filenames_pickle(test_path)
    else:
        print(f"File not found: {test_path}")
    
    # Inspect captions
    captions_path = os.path.join(oxford_dir, 'captions_DAMSM.pickle')
    if os.path.exists(captions_path):
        inspect_captions_pickle(captions_path)
    else:
        print(f"File not found: {captions_path}")
    
    # Check if total counts make sense
    print("\n==================== Summary ====================")
    train_count = 0
    test_count = 0
    captions_count = 0
    
    try:
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
            train_count = len(train_data)
    except:
        pass
        
    try:
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
            test_count = len(test_data)
    except:
        pass
        
    try:
        with open(captions_path, 'rb') as f:
            captions_data = pickle.load(f)
            captions_count = len(captions_data)
    except:
        pass
    
    print(f"Train filenames: {train_count}")
    print(f"Test filenames: {test_count}")
    print(f"Total images: {train_count + test_count}")
    print(f"Captions entries: {captions_count}")
    
    if captions_count != (train_count + test_count):
        print(f"Warning: Number of caption entries ({captions_count}) doesn't match total images ({train_count + test_count})")
    else:
        print("✓ Total counts consistent")

if __name__ == "__main__":
    main()
