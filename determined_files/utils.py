from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pprint import pprint
from ultralytics import YOLO
from ultralytics.yolo.data.build import load_inference_source
from ultralytics.yolo.utils import LINUX, ONLINE, ROOT, SETTINGS, yaml_load
from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, clean_url, colorstr, emojis, yaml_save)
from torch.optim import lr_scheduler
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
import time
from tqdm import tqdm
from torch.cuda import amp


def setup_scheduler():
    '''
    '''

def setup_dataloaders():
    '''
    '''

def setup_optimizer():
    '''
    '''
    
def setup_trainer():
    '''
    '''

def setup_train():
    '''
    '''
    
