import os
import sys
import argparse
import torch
from torchsummary import summary
import numpy as np

# Add the parent directory to the path to import the required modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.DAMSM import RNN_ENCODER, CNN_ENCODER

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect Oxford102 Encoders Architecture')
    parser.add_argument('--encoder_epoch', type=int, default=100,
                        help='Epoch of the encoder to load')
    args = parser.parse_args()
    return args

def create_dummy_args(encoder_epoch):
    """Create a dummy args object with minimal required attributes"""
    class Args:
        pass
    
    args = Args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.encoder_epoch = encoder_epoch
    args.data_dir = '../data/oxford102'
    
    # Set vocab size for Oxford102 (determined during training)
    args.vocab_size = 3082  # Your actual vocabulary size
    
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
        print(f"Vocabulary size: {encoder.encoder.num_embeddings}")

def main(args):
    print(f"Inspecting encoders for Oxford102 dataset, epoch {args.encoder_epoch}")
    
    # Create dummy args with necessary attributes
    dummy_args = create_dummy_args(args.encoder_epoch)
    
    # Try to locate and load the encoder files
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    img_encoder_path = os.path.join(base_dir, "data", "oxford102", 
                                    "DAMSMencoder", f"image_encoder{args.encoder_epoch}.pth")
    text_encoder_path = os.path.join(base_dir, "data", "oxford102", 
                                     "DAMSMencoder", f"text_encoder{args.encoder_epoch}.pth")
    
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
