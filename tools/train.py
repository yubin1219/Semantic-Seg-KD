import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from core.criterion import CrossEntropy, CriterionKD, CriterionCAT
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger

if args_s.seed > 0:
  import random
  print('Seeding with', args_s.seed)
  random.seed(args_s.seed)
  torch.manual_seed(args_s.seed)        

logger, final_output_dir, tb_log_dir = create_logger(args_s, args_s.cfg, 'train')
logger.info(pprint.pformat(args_s))

writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

# cudnn related setting
cudnn.benchmark = args_s.CUDNN.BENCHMARK
cudnn.deterministic = args_s.CUDNN.DETERMINISTIC
cudnn.enabled = args_s.CUDNN.ENABLED

# build model
model_S = get_seg_model(args_s).to(device)
model_T = get_seg_model(args_t).to(device)

batch_size = args_s.TRAIN.BATCH_SIZE_PER_GPU

# prepare data
crop_size = (args_s.TRAIN.IMAGE_SIZE[1], args_s.TRAIN.IMAGE_SIZE[0])
train_dataset = Cityscapes(
                        root=args_s.DATASET.ROOT,
                        list_path=args_s.DATASET.TRAIN_SET,
                        num_classes=args_s.DATASET.NUM_CLASSES,
                        multi_scale=args_s.TRAIN.MULTI_SCALE,
                        flip=args_s.TRAIN.FLIP,
                        ignore_label=args_s.TRAIN.IGNORE_LABEL,
                        crop_size=crop_size,
                        downsample_rate=args_s.TRAIN.DOWNSAMPLERATE,
                        scale_factor=args_s.TRAIN.SCALE_FACTOR)
trainloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=args_s.TRAIN.SHUFFLE,
        drop_last=True)

extra_epoch_iters = 0
if args_s.DATASET.EXTRA_TRAIN_SET:
  extra_train_dataset = Cityscapes(
                    root=args_s.DATASET.ROOT,
                    list_path=args_s.DATASET.EXTRA_TRAIN_SET,
                    num_classes=args_s.DATASET.NUM_CLASSES,
                    multi_scale=args_s.TRAIN.MULTI_SCALE,
                    flip=args_s.TRAIN.FLIP,
                    ignore_label=args_s.TRAIN.IGNORE_LABEL,
                    crop_size=crop_size,
                    downsample_rate=args_s.TRAIN.DOWNSAMPLERATE,
                    scale_factor=args_s.TRAIN.SCALE_FACTOR)

  extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=batch_size,
            shuffle=args_s.TRAIN.SHUFFLE,
            drop_last=True)
  
  extra_epoch_iters = np.int(extra_train_dataset.__len__() / 
                        args_s.TRAIN.BATCH_SIZE_PER_GPU)
  
test_size = (args_s.TEST.IMAGE_SIZE[1], args_s.TEST.IMAGE_SIZE[0])
test_dataset = Cityscapes(root=args_s.DATASET.ROOT,
                        num_samples=args_s.TEST.NUM_SAMPLES,
                        list_path=args_s.DATASET.TEST_SET,
                        num_classes=args_s.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=args_s.TRAIN.IGNORE_LABEL,
                        base_size=args_s.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args_s.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False)

# criterion
criterion_dsn = CrossEntropy(args_s, ignore_label=args_s.TRAIN.IGNORE_LABEL, weight=train_dataset.class_weights).to(device)
criterion_kd = CriterionKD().to(device)
criterion_cat = CriterionCAT().to(device)

# optimizer
optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model_S.parameters()), 'lr': args_s.TRAIN.LR}],
                                lr=args_s.TRAIN.LR,
                                momentum=args_s.TRAIN.MOMENTUM,
                                weight_decay=args_s.TRAIN.WD,
                                nesterov=args_s.TRAIN.NESTEROV
                                )

epoch_iters = np.int(train_dataset.__len__() / args_s.TRAIN.BATCH_SIZE_PER_GPU)
        
best_mIoU = 0
last_epoch = 0

checkpoint = torch.load('pretrained_models/hrnet_ocr_cs_8162_torch11.pth', map_location=device)
model_T.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint.items() if k.startswith('model.')})
model_T.eval()

if True:
  model_state_file = os.path.join('','checkpoint.pth.tar') ## 파일명 변경
  if os.path.isfile(model_state_file):
    checkpoint = torch.load(model_state_file, map_location=device)
    best_mIoU = checkpoint['best_mIoU']
    model_S.load_state_dict(checkpoint['state_dict'])

start = timeit.default_timer()
end_epoch = args_s.TRAIN.END_EPOCH + args_s.TRAIN.EXTRA_EPOCH
num_iters = args_s.TRAIN.END_EPOCH * epoch_iters
extra_iters = args_s.TRAIN.EXTRA_EPOCH * extra_epoch_iters

for epoch in range(last_epoch, end_epoch):
  current_trainloader = extra_trainloader if epoch >= args_s.TRAIN.END_EPOCH else trainloader
  if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
    current_trainloader.sampler.set_epoch(epoch)

  if epoch >= args_s.TRAIN.END_EPOCH:
    train(args_s, epoch, args_s.TRAIN.END_EPOCH, args_s.TRAIN.EXTRA_EPOCH, extra_epoch_iters, 
          args_s.TRAIN.EXTRA_LR, extra_iters, extra_trainloader, optimizer, model_S, model_T,
          criterion_dsn, criterion_cat, writer_dict)
  else:
    train(args_s, epoch, args_s.TRAIN.END_EPOCH, epoch_iters, args_s.TRAIN.LR,
          num_iters, trainloader, optimizer, model_S,model_T,
          criterion_dsn, criterion_cat, writer_dict) 

  valid_loss, mean_IoU, IoU_array = validate(args_s, testloader, model_S,
                                             criterion_dsn, writer_dict)

  if args_s.local_rank <= 0:
    logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
    if mean_IoU > best_mIoU:
      best_mIoU = mean_IoU
      torch.save(model_S.state_dict(), os.path.join(final_output_dir, 'best_{:4.0f}.pth'.format(best_mIoU*10000)))
      torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model_S.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint_best.pth.tar'))
    else:
      torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model_S.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

    msg = 'Loss: {:.3f}, MeanIoU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(valid_loss, mean_IoU, best_mIoU)
    logging.info(msg)
    logging.info(IoU_array)
