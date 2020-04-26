from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True, isAvg=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin, reduce=isAvg)
        self.K = num_instances
        self.use_semi = use_semi

    def forward(self, inputs, targets, epoch):
        n = inputs.size(0)
        P = n // self.K

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        if self.use_semi:
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i * self.K + j][mask[i * self.K + j] == 0]
                    for pair in range(j + 1, self.K):
                        ap = dist[i * self.K + j][i * self.K + pair]
                        dist_ap.append(ap.unsqueeze(0))
                        dist_an.append(neg_examples.min().unsqueeze(0))
        else:
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


class HoughTripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True, isAvg=True):
        super(HoughTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin, reduce=isAvg)
        self.K = num_instances
        self.use_semi = use_semi

    def forward(self, inputs, targets, epoch, hough_mask):
        n = inputs.size(0)
        P = n // self.K

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_p, mask_n = (hough_mask.cuda(), (~hough_mask).cuda())
        for i in range(n):
            mask_p[i] = mask[i] & mask_p[i]
            if mask_p[i].max() == 0:
                mask_p[i] = mask[i]
            mask_n[i] = (~mask[i]) | mask_n[i]
            if mask_n[i].max() == 0:
                mask_n[i] = ~mask[i]
        dist_ap, dist_an = [], []
        if self.use_semi:
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i * self.K + j][mask[i * self.K + j] == 0]
                    for pair in range(j + 1, self.K):
                        ap = dist[i * self.K + j][i * self.K + pair]
                        dist_ap.append(ap.unsqueeze(0))
                        dist_an.append(neg_examples.min().unsqueeze(0))
        else:
            for i in range(n):
                dist_ap.append(dist[i][mask_p[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask_n[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


class RankingTripletLoss(nn.Module):
    def __init__(self, margin):
        super(RankingTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, epoch, original_size, annotation):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        def gen_target(dist):
            y = dist.data.new()
            y.resize_as_(dist.data)
            y.fill_(1)
            return Variable(y)

        loss = 0
        for i in range(original_size):
            dist_ap = dist[i][original_size + 2 * i].unsqueeze(0)
            dist_an = dist[i][original_size + 2 * i + 1].unsqueeze(0)
            target = gen_target(dist_ap)
            loss += F.margin_ranking_loss(dist_an, dist_ap, target, margin=annotation[i] * self.margin)

        return loss / original_size


class SortedTripletLoss(nn.Module):
    def __init__(self, margin, isAvg=True):
        super(SortedTripletLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduce=isAvg)
        self.margin = margin

    def forward(self, inputs, targets, epoch):
        n = inputs.size(0)
        P = n // 3

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        dist_ap, dist_an = [], []
        for i in range(P):
            dist_ap.append(dist[3 * i][3 * i + 1].unsqueeze(0))
            dist_an.append(dist[3 * i][3 * i + 2].unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data + self.margin).sum() * 1. / y.size(0)
        return loss, prec