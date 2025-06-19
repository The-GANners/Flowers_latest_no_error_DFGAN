import os
import sys
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

# Add the parent directory to the path to import the required modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.perpare import prepare_models
from models.DAMSM import RNN_ENCODER, CNN_ENCODER

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect DAMSM Encoders Architecture')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset name: coco, bird')
    parser.add_argument('--encoder_epoch', type=int, default=100,
                        help='Epoch of the encoder to load')
    args = parser.parse_args()
    return args

def create_dummy_args(dataset, encoder_epoch):
    """Create a dummy args object with minimal required attributes"""
    class Args:
        pass
    
    args = Args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.encoder_epoch = encoder_epoch
    
    # Set paths based on dataset
    if dataset.lower() == 'coco':
        args.data_dir = '../data/coco'
        args.vocab_size = 27297  # This should be the vocabulary size for COCO
    elif dataset.lower() == 'bird':
        args.data_dir = '../data/birds'
        args.vocab_size = 5450  # This should be the vocabulary size for CUB
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Set other required args for encoder loading
    args.TEXT = Args()
    args.TEXT.EMBEDDING_DIM = 256
    args.TEXT.DAMSM_NAME = None  # This will be constructed in the prepare_models function
    args.multi_gpus = False
    args.local_rank = 0
    
    return args

def inspect_encoder(encoder, encoder_type):
    """Print the architecture of an encoder"""
    print(f"\n{'='*20} {encoder_type} Encoder Architecture {'='*20}")
    print(encoder)
    
    # Count number of parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print shapes of key layers if available
    if encoder_type == "Image":
        if hasattr(encoder, 'define_module'):
            print("\nLayers:")
            for name, module in encoder.named_children():
                print(f"{name}: {module}")
    elif encoder_type == "Text":
        print(f"\nEmbedding dimension: {encoder.nhidden}")
        print(f"RNN type: {encoder.rnn_type}")
        print(f"Number of layers: {encoder.nlayers}")
        print(f"Dropout: {encoder.drop_prob}")
        print(f"Bidirectional: {encoder.bidirectional}")

def main(args):
    print(f"Inspecting encoders for {args.dataset.upper()} dataset, epoch {args.encoder_epoch}")
    
    # Create dummy args with necessary attributes
    dummy_args = create_dummy_args(args.dataset, args.encoder_epoch)
    
    # Try to locate and load the encoder files
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    img_encoder_path = os.path.join(base_dir, "data", args.dataset.lower(), 
                                    "DAMSMencoder", f"image_encoder100.pth")
    text_encoder_path = os.path.join(base_dir, "data", args.dataset.lower(), 
                                     "DAMSMencoder", f"text_encoder100.pth")
    
    print(f"Looking for image encoder at: {img_encoder_path}")
    print(f"Looking for text encoder at: {text_encoder_path}")
    
    if not os.path.exists(img_encoder_path) or not os.path.exists(text_encoder_path):
        print(f"ERROR: Encoder files not found. Please check the paths.")
        return
    
    # Load image encoder
    print("\nLoading image encoder...")
    image_encoder = CNN_ENCODER(dummy_args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(img_encoder_path, map_location='cpu')
    if 'model' in state_dict:
        image_encoder.load_state_dict(state_dict['model'])
    else:
        image_encoder.load_state_dict(state_dict)
    
    # Load text encoder
    print("Loading text encoder...")
    text_encoder = RNN_ENCODER(dummy_args.vocab_size, nhidden=dummy_args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(text_encoder_path, map_location='cpu')
    if 'model' in state_dict:
        text_encoder.load_state_dict(state_dict['model'])
    else:
        text_encoder.load_state_dict(state_dict)
    
    # Inspect encoders
    inspect_encoder(image_encoder, "Image")
    inspect_encoder(text_encoder, "Text")
    
    print("\nDone!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
