import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
import pprint

import torch
from torchvision.utils import save_image,make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args
from lib.utils import load_model_opt,save_models,load_npz, params_count
from lib.perpare import prepare_dataloaders,prepare_models
from lib.modules import sample_one_batch as sample, test as test, train as train
from lib.datasets import get_fix_data


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/model/coco.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=15,
                        help='number of workers(default: 15)')
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--imsize', type=int, default=256,
                        help='input imsize')
    parser.add_argument('--batch_size', type=int, default=12,  # <-- set default to 12
                        help='batch size')
    parser.add_argument('--train', type=bool, default=True,
                        help='if train model')
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='resume epoch')
    parser.add_argument('--resume_model_path', type=str, default='model',
                        help='the model for resume training')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if multi-gpu training under ddp')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--encoder_epoch', type=int, default=100,
                        help='epoch of the DAMSM encoder to use')
    args = parser.parse_args()
    return args


def main(args):
    time_stamp = get_time_stamp()
    # Fix the attribute error - use DATASET_NAME if CONFIG_NAME is not available
    config_name = getattr(args, 'CONFIG_NAME', getattr(args, 'DATASET_NAME', 'flower'))
    stamp = '_'.join([str(args.model),str(args.stamp),str(config_name),str(args.imsize),time_stamp])
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(config_name), stamp)
    log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(str(config_name), 'train', stamp)))
    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(config_name), 'train', stamp)))
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
        mkdir_p(args.model_save_file)
        mkdir_p(args.img_save_dir)
    # prepare TensorBoard
    if (args.multi_gpus==True) and (get_rank() != 0):
        writer = None
    else:
        writer = SummaryWriter(log_dir)
    # prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    args.vocab_size = train_ds.n_words  # <-- This must be set BEFORE prepare_models
    image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    fixed_img, fixed_sent, fixed_z = get_fix_data(train_dl, valid_dl, text_encoder, args)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        fixed_grid = make_grid(fixed_img.cpu(), nrow=8, normalize=True)
        writer.add_image('fixed images', fixed_grid, 0)
        img_name = 'z.png'
        img_save_path = osp.join(args.img_save_dir, img_name)
        vutils.save_image(fixed_img.data, img_save_path, nrow=8, normalize=True)
    # prepare optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=0.0004, betas=(0.0, 0.9))

    # Quick save test: save models immediately and exit
    if os.environ.get("DFGAN_SAVE_TEST", "0") == "1":
        print("Running quick model save test...")
        save_models(netG, netD, netC, optimizerG, optimizerD, 0, args.multi_gpus, args.model_save_file)
        print(f"Model save test successful! Check: {args.model_save_file}/state_epoch_000.pth")
        return
    
    # Change npz_path to the correct file for flower dataset
    if hasattr(args, 'DATASET_NAME') and str(args.DATASET_NAME).lower() == 'flower':
        args.npz_path = r"C:\Users\nanda\OneDrive\Desktop\DF-GAN\data\flower\npz\flower_val256_FIDK0.npz"
        
    m1, s1 = load_npz(args.npz_path)
    # load from checkpoint
    strat_epoch = 1
    if args.resume_epoch!=1:
        strat_epoch = args.resume_epoch+1
        path = osp.join(args.resume_model_path, 'state_epoch_%03d.pth'%(args.resume_epoch))
        netG, netD, netC, optimizerG, optimizerD = load_model_opt(netG, netD, netC, optimizerG, optimizerD, path, args.multi_gpus)
    # print args
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        pprint.pprint(args)
        arg_save_path = osp.join(log_dir, 'args.yaml')
        save_args(arg_save_path, args)
        print("Start Training")
    # Start training
    test_interval,gen_interval,save_interval = args.test_interval,args.gen_interval,args.save_interval
    #torch.cuda.empty_cache()
    # Ensure max_epoch is set (default to 600 if missing)
    if not hasattr(args, 'max_epoch'):
        # Try to get from TRAIN section if present
        if hasattr(args, 'TRAIN') and hasattr(args.TRAIN, 'MAX_EPOCH'):
            args.max_epoch = args.TRAIN.MAX_EPOCH
        else:
            args.max_epoch = 600
    for epoch in range(strat_epoch, args.max_epoch, 1):
        if (args.multi_gpus==True):
            sampler.set_epoch(epoch)
        start_t = time.time()
        # training
        args.current_epoch = epoch
        torch.cuda.empty_cache()
        train(train_dl, netG, netD, netC, text_encoder, optimizerG, optimizerD, args)
        #torch.cuda.empty_cache()
        # save
        if epoch%save_interval==0:
            save_models(netG, netD, netC, optimizerG, optimizerD, epoch, args.multi_gpus, args.model_save_file)
        # sample
        if epoch%gen_interval==0:
            sample(fixed_z, fixed_sent, netG, args.multi_gpus, epoch, args.img_save_dir, writer)
        # end epoch
        # test
        if epoch%test_interval==0:
            torch.cuda.empty_cache()
            fid = test(valid_dl, text_encoder, netG, args.device, m1, s1, epoch, args.max_epoch, \
                        args.sample_times, args.z_dim, args.batch_size, args.truncation, args.trunc_rate)
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            if epoch%test_interval==0:
                writer.add_scalar('FID', fid, epoch)
                print('The %d epoch FID: %.2f'%(epoch,fid))
            end_t = time.time()
            print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
            print('*'*40)
        #torch.cuda.empty_cache()
        

if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # Force num_workers to 2 regardless of config file
    args.num_workers = 15
    # Force batch size to 12 regardless of config file
    args.batch_size = 32
    
    # Override config values as requested
    args.num_workers = 15
    args.test_interval = 5
    
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    
    # Handle CUDA setup safely
    if args.cuda:
        torch.cuda.manual_seed_all(args.manual_seed)
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Check if the specified GPU is valid
            gpu_count = torch.cuda.device_count()
            if args.gpu_id >= gpu_count:
                print(f"Warning: GPU {args.gpu_id} requested but only {gpu_count} GPUs available. Using GPU 0.")
                args.gpu_id = 0
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
            print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        else:
            print("CUDA is not available. Using CPU instead.")
            args.device = torch.device('cpu')
            args.cuda = False
    else:
        args.device = torch.device('cpu')
    
    args.local_rank = 0  # Always 0 for single GPU/CPU
    args.multi_gpus = False  # Force single GPU mode
    main(args)


