from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils.meters import AverageMeter
import numpy as np
import random


class BaseTrainer(object):
    def __init__(self, model, newDataSet, criterions, optimizer, print_freq=1):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizer = optimizer
        self.print_freq = print_freq

    def train(self, epoch):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            loss, prec1 = self._forward(inputs, targets, epoch)
            loss = loss.mean()
            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            # add gradient clip for lstm
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, epoch):
        raise NotImplementedError


class CoTeaching(object):
    def __init__(self, model, coModel, newDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # noise sample mining
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                inputsCNNB, tarCNNB = inputs[0][lossIdx], targets[lossIdx]
                inputsCNNB, tarCNNB = [inputsCNNB[:int(remRate * lossIdx.shape[0]), ...]], tarCNNB[:int(
                    remRate * lossIdx.shape[0])]
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(inputsCNNB, tarCNNB, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), tarCNNB.size(0))
                precisions.update(precCNNB, tarCNNB.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
            else:
                # update CNNA
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                inputsCNNA, tarCNNA = inputs[0][lossIdx], targets[lossIdx]
                inputsCNNA, tarCNNA = [inputsCNNA[:int(remRate * lossIdx.shape[0]), ...]], tarCNNA[:int(
                    remRate * lossIdx.shape[0])]
                # pure noise loss
                lossCNNA, precCNNA = self._forward(inputsCNNA, tarCNNA, epoch, self.modelA)
                lossCNNA = lossCNNA.mean()
                # update
                losses.update(lossCNNA.item(), tarCNNA.size(0))
                precisions.update(precCNNA, tarCNNA.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs)  # outputs=[x1,x2,x3]
        # new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[1], targets, epoch)  # feature
        # x2 triplet loss
        # loss_global, prec_global = self.criterions[1](outputs[2], targets, epoch)  # fc
        return loss_tri, prec_tri


class PastTrainer(BaseTrainer):
    def __init__(self, model, index_map, index, ranking, dataset, nums, newDataSet, criterions, optimizer, print_freq=1):
        super(PastTrainer, self).__init__(model, newDataSet, criterions, optimizer, print_freq)
        self.index_map = index_map
        self.index = index
        self.ranking = ranking
        self.dataset = dataset
        self.nums = nums
        self.original_size = 0
        self.annotation = []

    def _parse_data(self, inputs):
        imgs, fnames, pids, _, _ = inputs
        indexs = [self.index_map[fname] for fname in fnames]
        imgs2, pids2 = [], []
        self.original_size = len(fnames)
        self.annotation = []
        for i in range(self.original_size):
            index = indexs[i]
            pid = pids[i]
            num = self.nums[pid]

            pn_rank = [random.randint(1, num + 1), random.randint(num + 1, 2 * num + 1)]
            pos_index = self.index[index][pn_rank[0]]
            neg_index = self.index[index][pn_rank[1]]
            pos_input = self.dataset._get_single_item(pos_index)
            neg_input = self.dataset._get_single_item(neg_index)
            imgs2 += [pos_input[0].numpy(), neg_input[0].numpy()]
            fnames += [pos_input[1], neg_input[1]]
            pids2 += [pos_input[2], neg_input[2]]
            self.annotation.append((pn_rank[1] - pn_rank[0]) / num)

        imgs2 = torch.tensor(imgs2)
        pids2 = torch.tensor(pids2).long()
        imgs = torch.cat((imgs, imgs2))
        pids = torch.cat((pids, pids2))

        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs)  # outputs=[x1,x2,x3]
        # new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[1][:self.original_size], targets[:self.original_size], epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[2][:self.original_size], targets[:self.original_size], epoch)  # fc
        CTL = loss_tri + loss_global
        RTL = self.criterions[2](outputs[1], targets, epoch, self.original_size, self.annotation)
        loss_classification = self.criterions[3](outputs[1][:self.original_size], targets[:self.original_size], epoch)
        return CTL + RTL * 2 + loss_classification, prec_global


class ClassificationTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs)  # outputs=[x1,x2,x3]
        # new added by wc
        # x1 triplet loss
        # loss_tri, prec_tri = self.criterions[0](outputs[1], targets, epoch)  # feature
        # x2 triplet loss
        # loss_global, prec_global = self.criterions[1](outputs[2], targets, epoch)  # fc
        # CTL = loss_tri + loss_global
        loss_classification, prec_classification = self.criterions[3](outputs[1], targets, epoch, True)
        # return CTL + loss_classification, prec_global
        # return loss_tri + loss_classification, prec_tri
        return loss_classification, prec_classification

class ClassificationTrainer2(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs)  # outputs=[x1,x2,x3]
        # new added by wc
        # x1 triplet loss
        # loss_tri, prec_tri = self.criterions[0](outputs[1], targets, epoch)  # feature
        # x2 triplet loss
        # loss_global, prec_global = self.criterions[1](outputs[2], targets, epoch)  # fc
        # CTL = loss_tri + loss_global
        labels = targets.long().cuda()
        loss = nn.CrossEntropyLoss()(outputs[2], labels)
        precision = (outputs[2].argmax(1) == labels).sum().float() / labels.size(0)
        # return CTL + loss_classification, prec_global
        # return loss_tri + loss_classification, prec_tri
        return loss, precision


class RCoTeaching(object):
    """
    RCT implemention
    """

    def __init__(self, model, coModel, newDataSet, noiseDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            # noise data
            try:
                noiseInput = next(self.noiseData)
            except:
                noiseLoader = iter(self.noiseData)
                noiseInput = next(noiseLoader)
            noiseInput, noiseLab = self._parse_data(noiseInput)
            if i % 2 != 0:
                # update CNNA
                lossNoise, _ = self._forward(noiseInput, noiseLab, epoch, self.modelB)  # assigned samples
                lossPure, _ = self._forward(inputs, targets, epoch, self.modelB)
                # # assigned's easy samples
                lossIdx, lossPureIdx = np.argsort(lossNoise.data.cpu()).cuda(), np.argsort(lossPure.data).cuda()
                smallNoise = noiseInput[0][lossIdx[:int(remRate * lossNoise.shape[0])], ...]
                smallPure = inputs[0][lossPureIdx[:int(remRate * lossPure.shape[0])], ...]
                smallNoiseLab = noiseLab[lossIdx[:int(remRate * lossNoise.shape[0])]]
                smallPureLab = targets[lossPureIdx[:int(remRate * lossPure.shape[0])]]
                newLab = torch.cat([smallNoiseLab, smallPureLab])
                lossCNNA, precCNNA = self._forward([torch.cat([smallNoise, smallPure])], newLab, epoch, self.modelA)
                lossCNNA = lossCNNA.mean()
                losses.update(lossCNNA.item(), newLab.size(0))
                precisions.update(precCNNA, newLab.size(0))
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[0].step()
            else:
                # update CNNB
                lossNoise, _ = self._forward(noiseInput, noiseLab, epoch, self.modelA)  # assigned samples
                lossPure, _ = self._forward(inputs, targets, epoch, self.modelA)
                # # assigned's easy samples
                lossIdx, lossPureIdx = np.argsort(lossNoise.data.cpu()).cuda(), np.argsort(lossPure.data.cpu()).cuda()
                smallNoise = noiseInput[0][lossIdx[:int(remRate * lossNoise.shape[0])], ...]
                smallPure = inputs[0][lossPureIdx[:int(remRate * lossPure.shape[0])], ...]
                smallNoiseLab = noiseLab[lossIdx[:int(remRate * lossNoise.shape[0])]]
                smallPureLab = targets[lossPureIdx[:int(remRate * lossPure.shape[0])]]
                newLab = torch.cat([smallNoiseLab, smallPureLab])
                lossCNNB, precCNNB = self._forward([torch.cat([smallNoise, smallPure])], newLab, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), newLab.size(0))
                precisions.update(precCNNB, newLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class CoTrainerAsy(object):
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, newDataSet, noiseDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
            else:
                # update CNNA
                try:
                    noiseInput = next(self.noiseData)
                except:
                    noiseLoader = iter(self.noiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossCNNA = lossMix.mean()
                # update
                losses.update(lossCNNA.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class CoTrainerAsySep(object):
    def __init__(self, model, coModel, newDataSet, noiseDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]]
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
            else:
                # update CNNA
                try:
                    noiseInput = next(self.noiseData)
                except:
                    noiseLoader = iter(self.noiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # mix update, part assigned and part unassigned
                # mixInput, mixLab = [torch.cat([inputs[0],noiseInput])], torch.cat([targets,noiseLab])
                lossCNNAnoise, precCNNAnoise = self._forward([noiseInput], noiseLab, epoch, self.modelA)
                lossCNNApure, precCNNApure = self._forward(inputs, targets, epoch, self.modelA)
                lossCNNA = 0.1 * lossCNNAnoise.mean() + lossCNNApure.mean()
                # update
                losses.update(lossCNNA.item(), targets.size(0))
                precisions.update(precCNNApure, targets.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class EvoTrainer(object):
    def __init__(self, model, newDataSet, noiseDataSet, criterions, optimizer, print_freq=1):
        self.model = model
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizer = optimizer
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            # update CNNA
            lossPure, prec1 = self._forward(inputs, targets, epoch, self.model)  # assigned samples
            pureIdx = np.argsort(lossPure.data.cpu()).cuda()
            pureInput, targets = inputs[0][pureIdx], targets[pureIdx]
            pureInput, targets = pureInput[:int(remRate * lossPure.shape[0]), ...], targets[
                                                                                    :int(remRate * lossPure.shape[0])]
            # update CNNA
            try:
                noiseInput = next(noiseLoader)
            except:
                noiseLoader = iter(self.noiseData)
                noiseInput = next(noiseLoader)
            noiseInput, noiseLab = self._parse_data(noiseInput)
            lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.model)
            # sample mining
            lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
            noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
            noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                remRate * lossNoise.shape[0])]
            # mix update, part assigned and part unassigned
            mixInput, mixLab = [torch.cat([pureInput, noiseInput])], torch.cat([targets, noiseLab])
            lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.model)
            lossCNNA = lossMix.mean()
            # update
            losses.update(lossCNNA.item(), mixLab.size(0))
            precisions.update(precCNNA, mixLab.size(0))
            # update CNNA
            self.optimizer.zero_grad()
            lossCNNA.backward()
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue
            # update modelA
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global

class HoughTrainer(Trainer):
    def __init__(self, model, hough, newDataSet, criterions, optimizer, print_freq=1):
        super(HoughTrainer, self).__init__(model, newDataSet, criterions, optimizer, print_freq)
        self.hough = hough
        self.hough_mask = None

    def _parse_data(self, inputs):
        imgs, fname, pids, cams, times = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        self._gen_mask(cams, times)
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs)  # outputs=[x1,x2,x3]
        # new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch, self.hough_mask)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch, self.hough_mask)  # fc
        CTL = loss_tri + loss_global
        loss_classification = self.criterions[3](outputs[1], targets, epoch)
        return CTL + loss_classification, prec_global

    def _gen_mask(self, cams, times):
        length = len(cams)
#         hough_mask = torch.zeros((length, length))
#         for i in range(length):
#             for j in range(length):
#                 if self.hough.on_peak(cams[i], cams[j], times[i], times[j]):
#                     hough_mask[i][j] = 1
        hough_mask = torch.ones((length, length))
        self.hough_mask = hough_mask.eq(torch.ones((length, length)))
