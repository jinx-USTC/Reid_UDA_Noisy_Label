from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterions, print_freq=1):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq

    def train(self, epoch, data_loader, optimizer):
        self.model.train()
        # for name, param in self.model.named_parameters():
        #     if 'classifier' in name:
        #         param.requires_grad = False

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        #inputs_all = []
        #targets_all = []
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets, epoch)
            # # for My accumulated loss:
            # if (i + 1) % 20 == 0:
            #     inputs_all.append(*inputs) # '*' 代表序列解包
            #     targets_all.append(targets)
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            #add gradient clip for lstm
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

        # # for My accumulated loss
        # inputs_all = torch.cat(inputs_all, 0)
        # targets_all = torch.cat(targets_all, 0)
        # _, _, features_all = self._forward([inputs_all], targets_all, epoch)
        # loss_accumulated = self.criterions[2](features_all, targets_all)
        # optimizer.zero_grad()
        # loss_accumulated.backward()
        # # add gradient clip for lstm
        # for param in self.model.parameters():
        #     try:
        #         param.grad.data.clamp(-1., 1.)
        #     except:
        #         continue
        # optimizer.step()

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, epoch):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs) #outputs=[x1,x2,x3]
        #new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        #new added by xin:
        loss_accumulated = self.criterions[2](outputs[0], targets)

        return loss_tri+loss_global+loss_accumulated, prec_global




class BaseTrainer_with_learnable_label(object):
    def __init__(self, model, classifier, criterions, print_freq=1):
        super(BaseTrainer_with_learnable_label, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq
        self.classifier = classifier

    def train(self, epoch, data_loader, new_label, optimizer, optimizer_cla):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        import torch.nn as nn
        softmax = nn.Softmax(dim=1).cuda()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets, index = self._parse_data(inputs)

            index = index.numpy()
            new_label_update = new_label
            new_label_update = new_label_update[index,:]
            new_label_update = torch.FloatTensor(new_label_update)
            new_label_update = new_label_update.cuda(async = True)
            new_label_update = torch.autograd.Variable(new_label_update,requires_grad = True)
            # obtain label distributions (y_hat)
            last_updating_label_var = softmax(new_label_update)

            loss, prec1 = self._forward(inputs, targets, last_updating_label_var, epoch)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            optimizer_cla.zero_grad()
            loss.backward()
            #add gradient clip for lstm
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue

            optimizer.step()
            optimizer_cla.step()

            # update pseudo by back-propagation
            lambda1 = 600
            new_label_update.data.sub_(lambda1*new_label_update.grad.data)
            new_label[index,:] = new_label_update.data.cpu().numpy()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, last_updating_label_var, epoch):
        raise NotImplementedError


class Trainer_with_learnable_label(BaseTrainer_with_learnable_label):
    def _parse_data(self, inputs):
        imgs, _, pids, _, index = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets, index

    def _forward(self, inputs, targets, last_updating_label_var, epoch):
        outputs = self.model(*inputs) #outputs=[x1,x2,x3]
        #new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        #new added by xin:
        loss_accumulated = self.criterions[2](outputs[0], targets)

        # Noisy label updating:
        import torch.nn as nn
        logsoftmax = nn.LogSoftmax(dim=1).cuda()
        softmax = nn.Softmax(dim=1).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        output = self.classifier(outputs[0])
        loss_cla = torch.mean(softmax(output) * (logsoftmax(output) - torch.log((last_updating_label_var))))

        # lo is compatibility loss
        loss_lo = criterion(last_updating_label_var, targets)

        # le is entropy loss
        loss_le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

        return loss_tri + loss_global + loss_accumulated + (loss_cla + 0.4*loss_lo + 0.1*loss_le), prec_global
