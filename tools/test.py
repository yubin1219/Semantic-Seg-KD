from argparse import Namespace
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from datasets.cityscapes import Cityscapes
from models.hrnet_ocr_seg import get_seg_model
from utils.utils import get_confusion_matrix

def test(args_s, sv_dir=''):
  random.seed(args_s.seed)
  np.random.seed(args_s.seed)
  torch.manual_seed(args_s.seed)
  torch.cuda.manual_seed(args_s.seed)
  torch.cuda.manual_seed_all(args_s.seed)  
  
  # cudnn related setting
  cudnn.benchmark = args_s.CUDNN.BENCHMARK
  cudnn.deterministic = args_s.CUDNN.DETERMINISTIC
  cudnn.enabled = args_s.CUDNN.ENABLED
  
  device = "cuda" if torch.cuda.is_available() else 'cpu'
  model_S = get_seg_model(args_s).to(device)
  
  test_size = (args_s.TEST.IMAGE_SIZE[1], args_s.TEST.IMAGE_SIZE[0])
  test_dataset = Cityscapes(root=args_s.DATASET.ROOT,
                        num_samples=args_s.TEST.NUM_SAMPLES,
                        list_path=args_s.DATASET.TEST_SET,
                        num_classes=args_s.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=True,
                        ignore_label=255,
                        base_size=args_s.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

  testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args_s.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False)
  
  model_S.eval()
  
  if 'val' in args_s.DATASET.TEST_SET:
    confusion_matrix = np.zeros(
        (args_s.DATASET.NUM_CLASSES, args_s.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                args_s,
                model,
                image,
                scales=args_s.TEST.SCALE_LIST,
                flip=args_s.TEST.FLIP_TEST
                )

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=args_s.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                args_s.DATASET.NUM_CLASSES,
                255)


            sv_path = os.path.join(sv_dir, 'test_results')
            if not os.path.exists(sv_path):
              os.mkdir(sv_path)
            test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    print('MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
            pixel_acc, mean_acc))

  
  if 'test' in args_s.DATASET.TEST_SET:
    with torch.no_grad():
      for _, batch in enumerate(tqdm(testloader)):
        image, size, name = batch
        size = size[0]
        pred = test_dataset.multi_scale_inference(
                args_s,
                model_S,
                image,
                scales=args_s.TEST.SCALE_LIST,
                flip=args_s.TEST.FLIP_TEST
                )

        if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
          pred = F.interpolate(pred, size[-2:],mode='bilinear', align_corners=args_s.MODEL.ALIGN_CORNERS)
        

        sv_path = os.path.join(sv_dir, 'test_results')
        if not os.path.exists(sv_path):
          os.mkdir(sv_path)
        test_dataset.save_pred(pred, sv_path, name)


if __name__ == '__main__':
  args_S = {'AUTO_RESUME': False,
         'local_rank': -1,
         'seed' : 304,
         'cfg' : 'seg_hrnet_ocr_w18_train_512x1024.yaml',
         'CUDNN': CfgNode({'BENCHMARK': True, 'DETERMINISTIC': False, 'ENABLED': True}),
         'DATASET': CfgNode({'DATASET': 'cityscapes',
             'EXTRA_TRAIN_SET': '',
             'NUM_CLASSES': 19,
             'ROOT': 'data/',
             'TEST_SET': 'list/cityscapes/test.lst'}),
         'DEBUG': CfgNode({'DEBUG': False,
           'SAVE_BATCH_IMAGES_GT': False,
           'SAVE_BATCH_IMAGES_PRED': False,
           'SAVE_HEATMAPS_GT': False,
           'SAVE_HEATMAPS_PRED': False}),
         'MODEL': CfgNode({'ALIGN_CORNERS': True,
           'EXTRA': {'FINAL_CONV_KERNEL': 1,
                     'STAGE1': {'BLOCK': 'BOTTLENECK',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4],
                                'NUM_CHANNELS': [64],
                                'NUM_MODULES': 1,
                                'NUM_RANCHES': 1},
                     'STAGE2': {'BLOCK': 'BASIC',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4, 4],
                                'NUM_BRANCHES': 2,
                                'NUM_CHANNELS':[18, 36],
                                'NUM_MODULES': 1},
                     'STAGE3': {'BLOCK': 'BASIC',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4, 4, 4],
                                'NUM_BRANCHES': 3,
                                'NUM_CHANNELS': [18, 36, 72],
                                'NUM_MODULES': 4},
                     'STAGE4': {'BLOCK': 'BASIC',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4, 4, 4, 4],
                                'NUM_BRANCHES': 4,
                                'NUM_CHANNELS': [18, 36, 72, 144],
                                'NUM_MODULES': 3}},
           'NAME': 'seg_hrnet_ocr',
           'NUM_OUTPUTS': 2,
           'OCR': {'DROPOUT': 0.05,
                   'KEY_CHANNELS': 256,
                   'MID_CHANNELS': 512,
                   'SCALE': 1},
           'PRETRAINED': 'pretrained_models/best_model.pth'}),
         'OUTPUT_DIR': 'output',
         'PIN_MEMORY': True,
         'PRINT_FREQ': 10,
         'RANK': 0,
         'TEST':CfgNode( {'BASE_SIZE': 2048,
          'BATCH_SIZE_PER_GPU': 1,
          'FLIP_TEST': True,
          'IMAGE_SIZE': [2048, 1024],
          'MODEL_FILE': '',
          'MULTI_SCALE': False,
          'NUM_SAMPLES': None,
          'OUTPUT_INDEX': -1,
          'SCALE_LIST': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]}),
         'WORKERS': 1}
  test_opts = Namespace(**args_S)
  test(test_opts)
