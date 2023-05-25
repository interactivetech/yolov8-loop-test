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
from torch.distributed import init_process_group,destroy_process_group
import datetime
import torch.distributed as dist
def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model.
    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLOv8 model, else False.

    Raises:
        AssertionError: If the AMP checks fail, indicating anomalies with the AMP functionality on the system.
    """
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices

    def amp_allclose(m, im):
        """All close FP32 vs AMP results."""
        a = m(im, device=device, verbose=False)[0].boxes.data  # FP32 inference
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.data  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    f = ROOT / 'assets/bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if ONLINE else np.ones((640, 640, 3))
    prefix = colorstr('AMP: ')
    LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8 model...')
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO('yolov8n.pt'), im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8. Setting 'amp=True'.")
    except AssertionError:
        LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                       f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
        return False
    return True

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
                           imgsz=None,
                           data=None,
                           device=None,
                           epochs=None,
                           batch=None,
                           RANK=None,
                           world_size=None,
                           workers=None):
    '''
    '''
    cfg = yaml_load('/run/determined/workdir/ultralytics/yolo/cfg/default.yaml')
    # print("DEFAULT DICT")
    # pprint(dict(cfg))
    '''
    5/18/2023 (Andrew) : Bad hack
    We dont want DDP (hence RANK=-1) but we want to use CUDA for training
    Set device to 0 temporarily to initalize model correctly, then 
    reset back to -1 to not do DDP training
    '''
    if device == -1:
        # HACK: Keep rank -1 to prevent DDP from training, but allow GPU
        device=0
        cfg.update(dict(model=f'{MODEL_NAME}.pt', 
                imgsz = imgsz, 
                data=data, 
                device=device,
                epochs=epochs,
                batch=batch,
                workers=workers))
        trainer = DetectionTrainer(overrides=cfg)
        device=-1
    else:
        cfg.update(dict(model=f'{MODEL_NAME}.pt', 
                        imgsz = imgsz, 
                        data=data, 
                        device=device,
                        epochs=epochs,
                        batch=batch,
                        workers=workers))
        trainer = DetectionTrainer(overrides=cfg)
    print("--RANK: ",RANK)
    print("world_size: ",world_size)
    if world_size > 1:
                trainer._setup_ddp(world_size)
    # print("trainer.device: ",trainer.device)
    
    # print("trainer.args: ",trainer.args)

    #--setup_train
    """
    Builds dataloaders and optimizer on correct rank process.
    """
    # Model
    ckpt = torch.load(f'{MODEL_NAME}.pt')
    trainer.start_epoch = 0
    
    trainer.model = trainer.get_model(cfg=f'{MODEL_NAME}.yaml', weights=ckpt, verbose=RANK == -1)
    trainer.model.nc = trainer.data['nc']  # attach number of classes to model
    trainer.model.names = trainer.data['names']  # attach class names to model
    trainer.model.args = trainer.args  # attach hyperparameters to model
    if device == -1:
        trainer.model = trainer.model.to('cuda:0')
    else:
        trainer.model = trainer.model.to(trainer.device)
    # trainer.amp = torch.tensor(trainer.args.amp).to(trainer.device)  # True or False
    # Check AMP
    trainer.amp = torch.tensor(trainer.args.amp).to(trainer.device)  # True or False
    if trainer.amp and RANK in (-1, 0):  # Single-GPU and DDP
        callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
        trainer.amp = torch.tensor(check_amp(trainer.model), device=trainer.device)
        callbacks.default_callbacks = callbacks_backup  # restore callbacks
    if RANK > -1:  # DDP
        dist.broadcast(trainer.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
    trainer.amp = bool(trainer.amp)  # as boolean
    trainer.scaler = amp.GradScaler(enabled=trainer.amp)

    # trainer.data = yaml_load(trainer.args.data)
    # Check imgsz
    gs = max(int(trainer.model.stride.max() if hasattr(trainer.model, 'stride') else 32), 32)  # grid size (max stride)
    trainer.args.imgsz = check_imgsz(trainer.args.imgsz, stride=gs, floor=gs, max_dim=1)
    # Batch size
    return trainer
    
def setup_train(MODEL_NAME,
               imgsz=128,
               data=None,
               device=None,
               epochs=None,
               batch=None,
               RANK=None,
               world_size=None,
               workers=None):
    '''
    '''
    # RANK=-1
    trainer = setup_trainer_and_model(MODEL_NAME,
                                       imgsz=128,
                                       data=data,
                                       device=device,
                                       epochs=epochs,
                                       batch=batch,
                                       RANK=RANK,
                                       world_size=world_size,
                                       workers=workers)

    
    trainer.stopper, trainer.stop = EarlyStopping(patience=trainer.args.patience), False

    # dataloaders
    world_size = world_size  # TODO(ANDREW): default to device 0 # change for per gpu 
    # RANK=-1
    # ckpt = None
    batch_size = trainer.batch_size // world_size if world_size > 1 else trainer.batch_size
    trainer.batch_size = batch_size
    print("trainer.batch_size: ",trainer.batch_size)
    
    setup_dataloaders(trainer,batch_size,RANK)
    setup_optimizer(trainer)
    # Scheduler
    
    setup_scheduler(trainer)
    if RANK in (-1, 0):
        metric_keys = trainer.validator.metrics.keys + trainer.label_loss_items(prefix='val')
        trainer.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
        trainer.ema = ModelEMA(trainer.model)
        if trainer.args.plots and not trainer.args.v5loader:
            trainer.plot_training_labels()
    # trainer.resume_training(ckpt)
    
    trainer.scheduler.last_epoch = trainer.start_epoch - 1  # do not move
    return trainer, world_size, batch_size, RANK



    
