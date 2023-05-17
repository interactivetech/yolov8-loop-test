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

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n.pt'
CFG = 'yolov8n.yaml'
SOURCE = ROOT / 'assets/bus.jpg'
SOURCE_GREYSCALE = Path(f'{SOURCE.parent / SOURCE.stem}_greyscale.jpg')
SOURCE_RGBA = Path(f'{SOURCE.parent / SOURCE.stem}_4ch.png')

# Convert SOURCE to greyscale and 4-ch
im = Image.open(SOURCE)
im.convert('L').save(SOURCE_GREYSCALE)  # greyscale
im.convert('RGBA').save(SOURCE_RGBA)  # 4-ch PNG with alpha

# model = YOLO(CFG)
# model(SOURCE)
# args = dict(model='yolov8n.yaml', data='coco128.yaml', device='cpu')
cfg = yaml_load('/run/determined/workdir/ultralytics/ultralytics/yolo/cfg/default.yaml')
pprint(dict(cfg))
cfg.update(dict(model='yolov8n.pt', imgsz = 128, data='/run/determined/workdir/shared_fs/data/flir-camera-objects-yolo/data.yaml', device=0,epochs=2,batch=64,workers=8))
trainer = DetectionTrainer(overrides=cfg)
print("trainer.device: ",trainer.device)
# trainer.device = select_device(device='', batch=0)
print(trainer.args)

#--setup_train
"""
Builds dataloaders and optimizer on correct rank process.
"""
# Model
RANK=-1
ckpt = torch.load('yolov8n.pt')
trainer.model = trainer.get_model(cfg='yolov8n.yaml', weights=ckpt, verbose=RANK == -1)
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
print(trainer.batch_size)

# Optimizer
print("trainer.args.optimizer: ",trainer.args.optimizer)
accumulate = max(round(trainer.args.nbs / trainer.batch_size), 1)  # accumulate loss before optimizing
weight_decay = trainer.args.weight_decay * trainer.batch_size * accumulate / trainer.args.nbs  # scale weight_decay
trainer.optimizer = trainer.build_optimizer(model=trainer.model,
                                        name=trainer.args.optimizer,
                                        lr=trainer.args.lr0,
                                        momentum=trainer.args.momentum,
                                        decay=weight_decay)
# Scheduler

if trainer.args.cos_lr:
    trainer.lf = one_cycle(1, trainer.args.lrf, trainer.epochs)  # cosine 1->hyp['lrf']
else:
    trainer.lf = lambda x: (1 - x / trainer.epochs) * (1.0 - trainer.args.lrf) + trainer.args.lrf  # linear
trainer.scheduler = lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=trainer.lf)
trainer.stopper, trainer.stop = EarlyStopping(patience=trainer.args.patience), False

# dataloaders
world_size = 1  # TODO(ANDREW): default to device 0 # change for per gpu 
RANK=-1
ckpt = None
batch_size = trainer.batch_size // world_size if world_size > 1 else trainer.batch_size
trainer.train_loader = trainer.get_dataloader(trainer.trainset, batch_size=batch_size, rank=RANK, mode='train')
if RANK in (-1, 0):
    trainer.test_loader = trainer.get_dataloader(trainer.testset, batch_size=batch_size * 2, rank=-1, mode='val')
    trainer.validator = trainer.get_validator()
    metric_keys = trainer.validator.metrics.keys + trainer.label_loss_items(prefix='val')
    trainer.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
    trainer.ema = ModelEMA(trainer.model)
    if trainer.args.plots and not trainer.args.v5loader:
        trainer.plot_training_labels()
# trainer.resume_training(ckpt)
trainer.scheduler.last_epoch = trainer.start_epoch - 1  # do not move

## _do_train


trainer.epoch_time = None
trainer.epoch_time_start = time.time()
trainer.train_time_start = time.time()
nb = len(trainer.train_loader)  # number of batches
nw = max(round(trainer.args.warmup_epochs * nb), 100)  # number of warmup iterations
last_opt_step = -1
print(f'Image sizes {trainer.args.imgsz} train, {trainer.args.imgsz} val\n'
            f'Using {trainer.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', trainer.save_dir)}\n"
            f'Starting training for {trainer.epochs} epochs...')
if trainer.args.close_mosaic:
    base_idx = (trainer.epochs - trainer.args.close_mosaic) * nb
    trainer.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
for epoch in range(trainer.start_epoch, trainer.epochs):
    trainer.epoch = epoch
    trainer.model.train()
    # if RANK != -1:
    #     trainer.train_loader.sampler.set_epoch(epoch)
    pbar = enumerate(trainer.train_loader)
    # Update dataloader attributes (optional)
    if epoch == (trainer.epochs - trainer.args.close_mosaic):
        print('Closing dataloader mosaic')
        if hasattr(trainer.train_loader.dataset, 'mosaic'):
            trainer.train_loader.dataset.mosaic = False
        if hasattr(trainer.train_loader.dataset, 'close_mosaic'):
            trainer.train_loader.dataset.close_mosaic(hyp=trainer.args)

    if RANK in (-1, 0):
        print(trainer.progress_string())
        pbar = tqdm(enumerate(trainer.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
    trainer.tloss = None
    trainer.optimizer.zero_grad()
    for i, batch in pbar:
        # self.run_callbacks('on_train_batch_start')
        # Warmup
        ni = i + nb * epoch
        if ni <= nw:
            xi = [0, nw]  # x interp
            trainer.accumulate = max(1, np.interp(ni, xi, [1, trainer.args.nbs / trainer.batch_size]).round())
            for j, x in enumerate(trainer.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(
                    ni, xi, [trainer.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * trainer.lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [trainer.args.warmup_momentum, trainer.args.momentum])

        # Forward
        with torch.cuda.amp.autocast(trainer.amp):
            # """Preprocesses a batch of images by scaling and converting to float."""
            # batch['img'] = batch['img'].to(trainer.device, non_blocking=True).float() / 255
            # print("batch['img'] : ",batch['img'].dtype)
            batch = trainer.preprocess_batch(batch)
            preds = trainer.model(batch['img'])
            trainer.loss, trainer.loss_items = trainer.criterion(preds, batch)
            if RANK != -1:
                trainer.loss *= world_size
            trainer.tloss = (trainer.tloss * i + trainer.loss_items) / (i + 1) if trainer.tloss is not None \
                else trainer.loss_items

        # Backward
        trainer.scaler.scale(trainer.loss).backward()

        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
        if ni - last_opt_step >= trainer.accumulate:
            trainer.optimizer_step()
            last_opt_step = ni

        # Log
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        loss_len = trainer.tloss.shape[0] if len(trainer.tloss.size()) else 1
        losses = trainer.tloss if loss_len > 1 else torch.unsqueeze(trainer.tloss, 0)
        if RANK in (-1, 0):
            pbar.set_description(
                ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                (f'{epoch + 1}/{trainer.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
            # self.run_callbacks('on_batch_end')
            if trainer.args.plots and ni in trainer.plot_idx:
                trainer.plot_training_samples(batch, ni)

        # self.run_callbacks('on_train_batch_end')

    trainer.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(trainer.optimizer.param_groups)}  # for loggers

    trainer.scheduler.step()
    trainer.run_callbacks('on_train_epoch_end')

    if RANK in (-1, 0):

        # Validation
        trainer.ema.update_attr(trainer.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == trainer.epochs) or trainer.stopper.possible_stop

        if trainer.args.val or final_epoch:
            trainer.metrics, trainer.fitness = trainer.validate()
        trainer.save_metrics(metrics={**trainer.label_loss_items(trainer.tloss), **trainer.metrics, **trainer.lr})
        trainer.stop = trainer.stopper(epoch + 1, trainer.fitness)

        # Save model
        if trainer.args.save or (epoch + 1 == trainer.epochs):
            trainer.save_model()
            # trainer.run_callbacks('on_model_save')

    tnow = time.time()
    trainer.epoch_time = tnow - trainer.epoch_time_start
    trainer.epoch_time_start = tnow
    trainer.run_callbacks('on_fit_epoch_end')
    torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

    # Early Stopping
    if RANK != -1:  # if DDP training
        broadcast_list = [trainer.stop if RANK == 0 else None]
        dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
        if RANK != 0:
            trainer.stop = broadcast_list[0]
    if trainer.stop:
        break  # must break all DDP ranks

if RANK in (-1, 0):
    # Do final val with best.pt
    LOGGER.info(f'\n{epoch - trainer.start_epoch + 1} epochs completed in '
                f'{(time.time() - trainer.train_time_start) / 3600:.3f} hours.')
    trainer.final_eval()
    if trainer.args.plots:
        trainer.plot_metrics()
    trainer.run_callbacks('on_train_end')
    torch.cuda.empty_cache()
    trainer.run_callbacks('teardown')

##----
# model.train(data='coco8.yaml', epochs=1, imgsz=32,device='cpu',workers=0)
# model.val(data='coco8.yaml', imgsz=32,device='cpu')
# model(SOURCE)