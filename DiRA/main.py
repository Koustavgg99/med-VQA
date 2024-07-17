"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa)
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
import dataset_RAD
import dataset_SLAKE
import dataset_CLEF
import base_model
from train import train
import gc
gc.collect()
torch.cuda.empty_cache()

import utils

try:
    import _pickle as pickle
except:
    import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models/san_mevf',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=100,
                        help='the number of epoches')
    parser.add_argument('--lr', default=0.005, type=float, metavar='lr',
                        help='initial learning rate')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')

    # Choices of attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - support testing, gpu training or sampling
    parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Training with Dataset
    parser.add_argument('--use_RAD', action='store_true', default=False,
                        help='RAD')
    parser.add_argument('--RAD_dir', type=str, default=r'D:\PhD\Project\VQA_RAD',
                        help='RAD dir')
    parser.add_argument('--use_SLAKE', action='store_true', default=False, help='SLAKE')
    parser.add_argument('--SLAKE_dir', type=str, default=r'D:\PhD\Project\Slake\Slake1.0',
                        help='SLAKE dir')
    
    parser.add_argument('--use_CLEF', action='store_true', default=False, help='CLEF')
    parser.add_argument('--CLEF_dir', type=str, default=r'D:\PhD\Project\VQA_Dataset',
                        help='CLEF dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')
    parser.add_argument('--dir_feat_dim', default=2048, type=int,
                        help='visual feature dim')
    parser.add_argument('--GAP', action='store_true', default=False,
                        help='if Global average pooling is used')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')
    # DiR pretraining
    # image encoder
    parser.add_argument('--dir', action='store_true', default=False, help='DiR pretraining used')
    parser.add_argument('--weight', type=str, default='roco_128_200_r34_di_best_checkpoint.pth', help='path to pretrained weights')
    parser.add_argument('--weights', type=str, default='dira', help='specify the weights applicable else specify imagenet')
    parser.add_argument('--freeze_img_encoder', action='store_true', default=False, help='Freeze Image encoder')
    parser.add_argument('--layer',default=9,type=int, help='layer to freeze till')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture:(default:resnet50)')

    # Return args
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # create output directory and log file
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())
    # Set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # Load dictionary and RAD training dataset
    if args.use_RAD:
        print("Loading dictionary from: {}.".format(os.path.join(args.RAD_dir, 'dictionary.pkl')))
        dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
        train_dset = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
        eval_dset = dataset_RAD.VQAFeatureDataset('test', args, dictionary)

    if args.use_SLAKE:
        print("Loading dictionary from: {}.".format(os.path.join(args.SLAKE_dir, 'dictionary.pkl')))
        dictionary = dataset_SLAKE.Dictionary.load_from_file(os.path.join(args.SLAKE_dir, 'dictionary.pkl'))
        train_dset = dataset_SLAKE.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
        eval_dset = dataset_SLAKE.VQAFeatureDataset('validate', args, dictionary)
    
    if args.use_CLEF:
        print("Loading dictionary from: {}.".format(os.path.join(args.CLEF_dir, 'dictionary.pkl')))
        dictionary = dataset_CLEF.Dictionary.load_from_file(os.path.join(args.CLEF_dir, 'dictionary.pkl'))
        train_dset = dataset_CLEF.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
        eval_dset = dataset_CLEF.VQAFeatureDataset('val', args, dictionary)
        
    batch_size = args.batch_size
    # Create VQA model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args)
    optim = None
    epoch = 0
    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        model.to(device)
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1
        
    # create training dataloader
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)
    if args.use_RAD:
        eval_loader = None
    else:
        eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=utils.trim_collate)
    
    # training phase
    if args.freeze_img_encoder:
        count=0
        print(f"Total trainable parameters initially:{sum(p.numel() for p in model.parameters())}")
        
        for child in model.dir_v_emb.children():
            count+=1
            print(f"--------->child{count}")
            if count == args.layer or args.layer==0:
                break
            
            for param in child.parameters():
                param.requires_grad = False
            print(f"Parameters child{count} :{sum(p.numel() for p in child.parameters())}")
        
        print(f"Total trainable parameters:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    train(args, model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)