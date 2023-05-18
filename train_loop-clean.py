from pathlib import Path

from utils import setup_train
import time
from tqdm import tqdm
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, clean_url, colorstr, emojis, yaml_save)


import numpy as np
import torch
from PIL import Image
import argparse
import torch.distributed as dist
parser = argparse.ArgumentParser(description='')
parser.add_argument('--local_rank', default=-1,type=int,help='node rank for dist training')

def train(trainer, world_size, batch_size, RANK):
    '''
    ## _do_train
    '''
    
    trainer.epoch_time = None
    trainer.epoch_time_start = time.time()
    trainer.train_time_start = time.time()
    nb = len(trainer.train_loader)  # number of batches
    nw = max(round(trainer.args.warmup_epochs * nb), 100)  # number of warmup iterations
    last_opt_step = -1
    if RANK in (-1, 0):
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
        if RANK != -1:
            trainer.train_loader.sampler.set_epoch(epoch)
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
            # print("losses: ",[i.item() for i in losses])
            # print("box_loss   cls_loss   dfl_loss")
            if RANK in (-1, 0):
                # print("--RANK: ",RANK)
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
            print("metrics: ",trainer.metrics)
            # Images  Instances   P R mAP50  mAP50-95
            '''
            0 bicycle 2854 1018 0.13627635481182399 0.015345825758399443 0.013432724878911876 0.0061077860924923545
            1 car 2854 11100 0.4345758515278086 0.2981981981981982 0.2847026227029621 0.13878515318645318
            2 dog 2854 56 1.0 0.0 0.0 0.0
            3 person 2854 9379 0.36676974626931835 0.153747734300032 0.13889793836405878 0.04776205043115383
            '''
            val_dict = {}
            val_dict.update(trainer.metrics)
            for i, c in enumerate(trainer.validator.metrics.ap_class_index):
                val_dict[f"{trainer.validator.names[c]}_P"]=trainer.validator.metrics.class_result(i)[0]
                val_dict[f"{trainer.validator.names[c]}_R"]=trainer.validator.metrics.class_result(i)[1]
                val_dict[f"{trainer.validator.names[c]}_mAP50"]=trainer.validator.metrics.class_result(i)[2]
                val_dict[f"{trainer.validator.names[c]}_mAP"]=trainer.validator.metrics.class_result(i)[3]
                # print(i,trainer.validator.names[c], trainer.validator.seen, trainer.validator.nt_per_class[c], trainer.validator.metrics.class_result(i))
            print("val_dict: ",val_dict)
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

def main(local_rank,world_size):
    '''
    '''
    MODEL_NAME='yolov8s'
    
    trainer, world_size, batch_size, RANK = setup_train(
                           MODEL_NAME,
                           imgsz=224,
                           data='/run/determined/workdir/shared_fs/andrew-demo-revamp/flir-camera-objects-yolo/data.yaml',
                           device=local_rank if local_rank !=-1 else 0,
                           epochs=2,
                           batch=32,
                           RANK=local_rank,
                           world_size = world_size if world_size>1 else 1,
                           workers=8)
    train(trainer, world_size, batch_size, RANK)
    

if __name__ == '__main__':
    args= parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    if args.local_rank == -1:
        # Hack, if running normal loop, ignore torch's count of GPUs
        world_size=1
    else:
        world_size = args.nprocs
    print(args.local_rank,args.nprocs)
    main(args.local_rank,world_size)