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


def setup_scheduler(trainer):
    '''
    '''
    if trainer.args.cos_lr:
        trainer.lf = one_cycle(1, trainer.args.lrf, trainer.epochs)  # cosine 1->hyp['lrf']
    else:
        trainer.lf = lambda x: (1 - x / trainer.epochs) * (1.0 - trainer.args.lrf) + trainer.args.lrf  # linear
    trainer.scheduler = lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=trainer.lf)

def setup_dataloaders(trainer,batch_size,RANK):
    '''
    '''
    trainer.train_loader = trainer.get_dataloader(trainer.trainset, batch_size=batch_size, rank=RANK, mode='train')
    if RANK in (-1, 0):
        trainer.test_loader = trainer.get_dataloader(trainer.testset, batch_size=batch_size * 2, rank=-1, mode='val')
        trainer.validator = trainer.get_validator()

def setup_optimizer(trainer):
    '''
    '''
    # Optimizer
    print("trainer.args.optimizer: ",trainer.args.optimizer)
    trainer.accumulate = max(round(trainer.args.nbs / trainer.batch_size), 1)  # accumulate loss before optimizing
    weight_decay = trainer.args.weight_decay * trainer.batch_size * trainer.accumulate / trainer.args.nbs  # scale weight_decay
    trainer.optimizer = trainer.build_optimizer(model=trainer.model,
                                            name=trainer.args.optimizer,
                                            lr=trainer.args.lr0,
                                            momentum=trainer.args.momentum,
                                            decay=weight_decay)
    
def setup_trainer_and_model(MODEL_NAME,
                           imgsz=128,
                           data='/run/determined/workdir/shared_fs/andrew-demo-revamp/flir-camera-objects-yolo/data.yaml',
                           device=None,
                           epochs=None,
                           batch=None,
                           workers=None):
    '''
    '''
    cfg = yaml_load('/run/determined/workdir/yolov8-loop-test/ultralytics/yolo/cfg/default.yaml')
    # print("DEFAULT DICT")
    # pprint(dict(cfg))
    cfg.update(dict(model=f'{MODEL_NAME}.pt', 
                    imgsz = imgsz, 
                    data=data, 
                    device=device,
                    epochs=epochs,
                    batch=batch,
                    workers=workers))
    trainer = DetectionTrainer(overrides=cfg)
    # print("trainer.device: ",trainer.device)
    
    # print("trainer.args: ",trainer.args)

    #--setup_train
    """
    Builds dataloaders and optimizer on correct rank process.
    """
    # Model

    ckpt = torch.load(f'{MODEL_NAME}.pt')
    trainer.model = trainer.get_model(cfg=f'{MODEL_NAME}.yaml', weights=ckpt, verbose=RANK == -1)
    trainer.model.nc = trainer.data['nc']  # attach number of classes to model
    trainer.model.names = trainer.data['names']  # attach class names to model
    trainer.model.args = trainer.args  # attach hyperparameters to model
    trainer.model = trainer.model.to(trainer.device)
    # trainer.amp = torch.tensor(trainer.args.amp).to(trainer.device)  # True or False
    trainer.amp = True
    trainer.scaler = amp.GradScaler(enabled=trainer.amp)

    # trainer.data = yaml_load(trainer.args.data)
    # Check imgsz
    gs = max(int(trainer.model.stride.max() if hasattr(trainer.model, 'stride') else 32), 32)  # grid size (max stride)
    trainer.args.imgsz = check_imgsz(trainer.args.imgsz, stride=gs, floor=gs, max_dim=1)
    # Batch size
    print("trainer.batch_size: ",trainer.batch_size)
    return trainer
    
def setup_train(MODEL_NAME,
               imgsz=128,
               data='/run/determined/workdir/shared_fs/andrew-demo-revamp/flir-camera-objects-yolo/data.yaml',
               device=None,
               epochs=None,
               batch=None,
               workers=None):
    '''
    '''
    RANK=-1
    trainer = setup_trainer_and_model(MODEL_NAME,
                                       imgsz=128,
                                       data='/run/determined/workdir/shared_fs/andrew-demo-revamp/flir-camera-objects-yolo/data.yaml',
                                       device=device,
                                       epochs=epochs,
                                       batch=batch,
                                       workers=workers)

    setup_optimizer(trainer)
    # Scheduler

    setup_scheduler(trainer)
    trainer.stopper, trainer.stop = EarlyStopping(patience=trainer.args.patience), False

    # dataloaders
    world_size = 1  # TODO(ANDREW): default to device 0 # change for per gpu 
    RANK=-1
    # ckpt = None
    batch_size = trainer.batch_size // world_size if world_size > 1 else trainer.batch_size
    setup_dataloaders(trainer,batch_size,RANK)
    
    if RANK in (-1, 0):
        metric_keys = trainer.validator.metrics.keys + trainer.label_loss_items(prefix='val')
        trainer.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
        trainer.ema = ModelEMA(trainer.model)
        if trainer.args.plots and not trainer.args.v5loader:
            trainer.plot_training_labels()
    # trainer.resume_training(ckpt)
    
    trainer.scheduler.last_epoch = trainer.start_epoch - 1  # do not move
    return trainer, world_size, batch_size, RANK



    
