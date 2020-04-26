#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import
import argparse
import time
import os.path as osp
import os 
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss, ClassificationLoss, RankingTripletLoss, HoughTripletLoss, SortedTripletLoss
from reid.trainers import PastTrainer, ClassificationTrainer, HoughTrainer, Trainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
import torch.nn.functional as F
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler, TripletSampler
from reid.utils.serialization import load_checkpoint, save_checkpoint

from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
#from h2o4gpu.cluster import KMeans
#from h2o4gpu.preprocessing import normalize
from reid.rerank import re_ranking, re_ranking2
from reid.st_model import ST_Model


def get_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training and validation images in target dataset
    train_set = dataset.trainval 
    num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=transformer),
        batch_size=batch_size//2, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, extfeat_loader, test_loader


def get_source_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training images on source dataset
    train_set = dataset.train
    num_classes = dataset.num_train_ids

    transformer = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    # get source data
    src_dataset, src_extfeat_loader = \
        get_source_data(args.src_dataset, args.data_dir, args.height,
                        args.width, args.batch_size, args.workers)
    # get target data
    tgt_dataset, num_classes, tgt_extfeat_loader, test_loader = \
        get_data(args.tgt_dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    # Hacking here to let the classifier be the number of source ids
    if args.src_dataset == 'dukemtmc':
        model = models.create(args.arch, num_classes=632, pretrained=False)
    elif args.src_dataset == 'market1501':
        model = models.create(args.arch, num_classes=676, pretrained=False)
    else:
        raise RuntimeError('Please specify the number of classes (ids) of the network.')

    # Load from checkpoint
    if args.resume:
        print('Resuming checkpoints from finetuned model on another dataset...\n')
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise RuntimeWarning('Not using a pre-trained model.')
    model = nn.DataParallel(model).cuda()

    # evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    # if args.evaluate: return

    # Criterion
    criterion = [
        SortedTripletLoss(args.margin, isAvg=True).cuda()
    ]


    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )


    # training stage transformer on input images
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((args.height,args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])

    evaluator = Evaluator(model, print_freq=args.print_freq)
    evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)

    st_model = ST_Model(tgt_dataset.meta['num_cameras'])

    # # Start training
    for iter_n in range(args.iteration):
        if args.lambda_value == 0:
            source_features = 0
        else:
            # get source datas' feature
            source_features, _ = extract_features(model, src_extfeat_loader, print_freq=args.print_freq)
            # synchronization feature order with src_dataset.train
            source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _, _ in src_dataset.train], 0)

        # extract training images' features
        print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n+1))
        target_features, tarNames = extract_features(model, tgt_extfeat_loader, print_freq=args.print_freq)
        # synchronization feature order with dataset.train
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _, _ in tgt_dataset.trainval], 0)
        # target_real_label = np.asarray([tarNames[f].unsqueeze(0) for f, _, _, _ in tgt_dataset.trainval])

        # calculate distance and rerank result
        target_features = target_features.numpy()
        rerank_dist = re_ranking(source_features, target_features, lambda_value=args.lambda_value)

        # if iter_n > 0:
        #     rerank_dist = st_model.apply(rerank_dist, tgt_dataset.trainval, tgt_dataset.trainval)

        cluster = HDBSCAN(metric='precomputed', min_samples=10)
        # select & cluster images as training set of this epochs
        clusterRes = cluster.fit(rerank_dist)
        labels, label_num = clusterRes.labels_, clusterRes.labels_.max() + 1
        # centers = np.zeros((label_num, target_features.shape[1]))
        # nums = [0] * target_features.shape[1]
        print('clusters num =', label_num)

        # generate new dataset
        new_dataset = []
        index = -1
        for (fname, _, cam, timestamp), label in zip(tgt_dataset.trainval, labels):
            index += 1
            if label == -1: continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, cam, timestamp))
            # centers[label] += target_features[index]
            # nums[label] += 1
        print('Iteration {} have {} training images'.format(iter_n+1, len(new_dataset)))

        # learn ST model
        same, _ = st_model.fit(new_dataset)
        # st_model.fit(tgt_dataset.trainval)

        def filter(i, j):
            _, _, c1, t1 = tgt_dataset.trainval[i]
            _, _, c2, t2 = tgt_dataset.trainval[j]
            return same.in_peak(c1, c2, t1, t2, 0.2)

        ranking = np.argsort(rerank_dist)[:, 1:]

        cluster_size = 23.535612535612536
        must_conn = int(cluster_size / 2)
        might_conn = int(cluster_size * 2)

        length = len(tgt_dataset.trainval)
        pos = [[] for _ in range(length)]
        neg = [[] for _ in range(length)]
        for i in range(length):
            for j_ in range(might_conn):
                j = ranking[i][j_]
                if j_ < must_conn and i in ranking[j][:must_conn]:
                    pos[i].append(j)
                elif i in ranking[j][:might_conn] and filter(i, j):
                    pos[i].append(j)
                # if j_ < must_conn or filter(i, j):
                #     pos[i].append(j)
                else:
                    neg[i].append(j)
            if len(neg[i]) < len(pos[i]):
                neg[i].extend(ranking[i][j_+1:j_+1+len(pos[i])-len(neg[i])])

        # learn visual model
        # for i in range(label_num):
        #     centers[i] /= nums[i]
        # criterion[3] = ClassificationLoss(normalize(centers, axis=1)).cuda()
        #
        # classOptimizer = torch.optim.Adam([
        #     {'params': model.parameters()},
        #     {'params': criterion[3].classifier.parameters(), 'lr': 1e-3}
        # ], lr=args.lr)


        train_loader = DataLoader(
            Preprocessor(tgt_dataset.trainval, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=TripletSampler(tgt_dataset.trainval, pos, neg),
            pin_memory=True, drop_last=True
        )

        trainer = Trainer(model, train_loader, criterion, optimizer)

        for epoch in range(args.epochs):
            trainer.train(epoch)

        rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)

    # Evaluate
    rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1, 'best_top1': rank_score.market1501[0],
        }, True, fpath=osp.join(args.logs_dir, 'adapted.pth.tar'))
    return (rank_score.map, rank_score.market1501[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('--src_dataset', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('--tgt_dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--noiseLam', type=float, default=0.5)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=models.names())
    # loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin of the triplet loss, default: 0.3")
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    # optimizer
    parser.add_argument('--lr', type=float, default=5e-5,
                        help="learning rate of all parameters")
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default = '')
    parser.add_argument('--evaluate', type=int, default=0,
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='')

    args = parser.parse_args()
    mean_ap, rank1 = main(args)
    results_file = np.asarray([mean_ap, rank1])
    file_name = time.strftime("%H%M%S", time.localtime())
    file_name = osp.join(args.logs_dir, file_name)
    np.save(file_name, results_file)
