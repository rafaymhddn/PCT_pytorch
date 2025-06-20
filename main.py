import argparse
import os
import time

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ModelNet40
from dataset_pickplace import PickPlaceDataset, get_dataloaders
from model import NaivePCTCls, SPCTCls, PCTCls, NaivePCTSeg
from util import cal_loss, Logger


models = {'navie_pct': NaivePCTCls,
          'spct': SPCTCls,
          'pct': PCTCls,
          'NaivePCTSeg': NaivePCTSeg}


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=2,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=2,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    data_root = 'data/pick_place'
    train_loader, val_loader, test_loader = get_dataloaders(data_root)
    

    device = args.device
    print(device)

    model = models[args.model]().to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        count = 0.0  # numbers of data
        model.train()
        train_pred = []
        train_true = []
        idx = 0  # iterations
        total_time = 0.0
        for data, label in (train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            
        print ('train total time is',total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print ('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
        
        scheduler.step()


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("mps" if args.cuda else "cpu")

    model = models[args.model]().to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1] 
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pct', choices=['navie_pct', 'spct', 'pct', 'NaivePCTSeg'],
                        help='which model you want to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_(args)

    io = Logger('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    # Set device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        io.cprint(f'Using GPU : {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
        torch.cuda.manual_seed(args.seed)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        args.device = torch.device("mps")
        io.cprint("Using Apple MPS (Metal Performance Shaders) backend")
        torch.manual_seed(args.seed)
    else:
        args.device = torch.device("cpu")
        io.cprint("Using CPU")
        torch.manual_seed(args.seed)

    # Now run training or evaluation
    if not args.eval:
        train(args, io)
    else:
        test(args, io)

