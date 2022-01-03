import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model_S, model_T, 
          criterion_dsn, criterion_cat, writer_dict):
    # Training
    model_S.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader):
        optimizer.zero_grad()
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        output_s = model_S(images)

        with torch.no_grad():
          output_t = model_T(images)

        loss_dsn = criterion_dsn(output_s[:2], labels.detach())
        loss_cat_feat = 0.4 * criterion_cat(output_s[2], output_t[2].detach(), output_t[1].detach())
        #loss_at_feat = 8 * criterion_at(output_s[2], output_t[2].detach())
        #loss_kd = 0.8 * criterion_kd(output_s[1], output_t[1].detach())

        #loss_cwd_feat = 50 * criterion_cwd(output_s[2], output_t[2].detach())
        #loss_cwd_logit = 4 * criterion_cwd(output_s[1], output_t[1].detach())

        loss = loss_dsn + loss_cat_feat

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters, power=0.8)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.5f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)
            msg2 = "loss_dsn: {:.4f}, loss_cat_feat: {:.4f}".format(loss_dsn, loss_cat_feat)
            logging.info(msg2)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
    
def validate(config, testloader, model, criterion, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            pred_s = model(image)[:2]
            loss = criterion(pred_s, label)

            if not isinstance(pred_s, (list, tuple)):
                pred_s = [pred_s]

            for i, x in enumerate(pred_s):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    
    return ave_loss.average(), mean_IoU, IoU_array
