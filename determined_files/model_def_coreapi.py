
import determined as det
from utils import *
from pathlib import Path

from utils import setup_train
import time
from tqdm import tqdm
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.downloads import attempt_download_asset
import torch.nn as nn

import numpy as np
import torch
from PIL import Image
import argparse
import torch.distributed as dist
import os
from ultralytics import YOLO

def is_parallel(model):
    '''
    return if model is DataParallel (DP) or DistributedDataParallel (DDP)
    '''
    return type(model) in (nn.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallelize_model(model):
    '''
    De-parallelize model: return a single GPU model if model is of type DP or DDP
    '''
    return model.module if is_parallel(model) else model

def load_state(latest_checkpoint, trial_id,core_context):
    '''
    '''
    with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            ckpt = torch.load(os.path.join(path,'last.pt'),map_location='cpu')
            model_ckpt = ckpt['model']
            epoch= ckpt['epoch']
            # self.epoch = ckpt['epoch']+1
            # self.losses = ckpt['losses']
            # self.val_losses = ckpt['val_losses']
            # self.train_accs = ckpt['train_accs']
            # self.val_accs = ckpt['val_accs']
            # self.model.load_state_dict(ckpt['model'])#ToDo load DP or DDP
            # self.model.to(self.gpu_id)
            # # get optimizer
            # self.optimizer = get_optimizer(self.model)
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            # print("Resuming Epoch {}".format(self.epoch))
            return model_ckpt, epoch

def save_state(model,
               core_context, 
               trial_id, 
               epochs_completed):
    '''
    '''
    ckpt_metadata = {"steps_completed":epochs_completed+1}
        # 8. Save checkpoint
    with core_context.checkpoint.store_path(ckpt_metadata) as (checkpoint_directory,uuid):
        ckpt = {
            'epoch': epochs_completed,
            'model': de_parallelize_model(model).state_dict()
        }
        torch.save(ckpt,os.path.join(checkpoint_directory,'last.pt'))
    del ckpt    


def test():
    '''
    '''




def train(core_context,
          trainer, 
          world_size, 
          batch_size, 
          RANK,
          trial_id):
    '''
    ## _do_train
    '''
    for op in core_context.searcher.operations():
        trainer.epochs = op.length
        trainer.epoch_time = None
        trainer.epoch_time_start = time.time()
        trainer.train_time_start = time.time()
        nb = len(trainer.train_loader)  # number of batches
        nw = max(round(trainer.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        if RANK in (-1, 0):
            steps = 0
            print(f'Image sizes {trainer.args.imgsz} train, {trainer.args.imgsz} val\n'
                        f'Using {trainer.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', trainer.save_dir)}\n"
                        f'Starting training for {trainer.epochs} epochs...')
        if trainer.args.close_mosaic:
            base_idx = (trainer.epochs - trainer.args.close_mosaic) * nb
            trainer.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    
        for epoch in range(trainer.start_epoch,op.length):
        # for epoch in range(trainer.start_epoch, trainer.epochs):
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
                if RANK in (-1, 0):
                    # print("--RANK: ",RANK)
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{trainer.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    loss_dict = {
                                 'box_loss':losses[0].item(),
                                 'cls_loss': losses[1].item(),
                                 'dfl_loss': losses[2].item()
                                 }
                    if steps%20 == 0:
                        # print("steps -- : ", steps)
                        core_context.train.report_training_metrics(
                            steps_completed=steps,
                            metrics=loss_dict,
                        )
                    steps+=1
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
                val_dict = {}
                val_dict.update(trainer.metrics)
                for i, c in enumerate(trainer.validator.metrics.ap_class_index):
                    val_dict[f"{trainer.validator.names[c]}_P"]=trainer.validator.metrics.class_result(i)[0]
                    val_dict[f"{trainer.validator.names[c]}_R"]=trainer.validator.metrics.class_result(i)[1]
                    val_dict[f"{trainer.validator.names[c]}_mAP50"]=trainer.validator.metrics.class_result(i)[2]
                    val_dict[f"{trainer.validator.names[c]}_mAP"]=trainer.validator.metrics.class_result(i)[3]
                    # print(i,trainer.validator.names[c], trainer.validator.seen, trainer.validator.nt_per_class[c], trainer.validator.metrics.class_result(i))
                # print("val_dict: ",val_dict)
                core_context.train.report_validation_metrics(
                            steps_completed=epoch+1,
                            metrics=val_dict,
                        )
                # Save model
                if trainer.args.save or (epoch + 1 == trainer.epochs):
                    trainer.save_model()
                    # trainer.run_callbacks('on_model_save')
                
                save_state(trainer.model,
                   core_context, 
                   trial_id, 
                   epoch)
                # NEW: Report progress only on rank 0.
                op.report_progress(epoch)
                # 7b.check for pre-emption signal
            if core_context.preempt.should_preempt():
                print("Preemption Signal Detected, stopping training...")
                return
                

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
            # Report completed only on rank 0.
            op.report_completed(val_dict['metrics/mAP50-95(B)'])

##----
# model.train(data='coco8.yaml', epochs=1, imgsz=32,device='cpu',workers=0)
# model.val(data='coco8.yaml', imgsz=32,device='cpu')
# model(SOURCE)

def main(local_rank,
         world_size,
         hparams,
         latest_checkpoint,
         trial_id):
    '''
    '''
    MODEL_NAME=hparams['model_name']
    attempt_download_asset(MODEL_NAME+'.pt')
    dname_to_path = {
     'x-ray-rheumatology':'/run/determined/workdir/shared_fs/andrew-demo-revamp/x-ray-rheumatology/data.yaml',
     'flir-camera-objects': '/run/determined/workdir/shared_fs/andrew-demo-revamp/flir-camera-objects/data.yaml'
    }
    trainer, world_size, batch_size, RANK = setup_train(
                           MODEL_NAME,
                           imgsz=hparams['imgsz'],
                           data=dname_to_path[hparams['dataset_name']],
                           device=local_rank if local_rank !=-1 else -1,
                           epochs=hparams['epochs'],
                           batch=hparams['global_batch_size'],
                           RANK=local_rank,
                           world_size = world_size if world_size>1 else 1,
                           workers=hparams['workers'])
    # At this point, nccl is initalized
    if RANK != -1:
        distributed = det.core.DistributedContext.from_torch_distributed()
        with det.core.init(distributed=distributed) as core_context:
            print("core_context.distributed.size: ",core_context.distributed.size)
            print("core_context.distributed.rank: ",core_context.distributed.rank)
            if latest_checkpoint is not None:
                print("Loading Checkpoint")
                model_ckpt, epoch = load_state(latest_checkpoint, trial_id,core_context)
                # print("model_ckpt keys: ",model_ckpt.keys())
                trainer.start_epoch = epoch
                # trainer.model = trainer.model.load_state_dict(model_ckpt)
                try:
                    trainer.model.load_state_dict(model_ckpt)
                except Exception:
                    # If the checkpointed model is non-DDP and the current model is DDP, append
                    # module prefix to the checkpointed data
                    if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
                        print("Loading non-DDP checkpoint into a DDP model.")
                        self._add_prefix_in_state_dict_if_not_present(model_ckpt, "module.")
                    else:
                        # If the checkpointed model is DDP and if we are currently running in
                        # single-slot mode, remove the module prefix from checkpointed data
                        print("Loading DDP checkpoint into a non-DDP model.")
                        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                            model_ckpt, "module."
                        )
                    trainer.model.load_state_dict(model_ckpt)
                print("Model Loaded")
            train(core_context,
                  trainer, 
                  core_context.distributed.size, 
                  batch_size, 
                  core_context.distributed.rank,
                  trial_id)
    else:
        
        with det.core.init(distributed=None) as core_context:
            # print("core_context.distributed.size: ",core_context.distributed.size)
            # print("core_context.distributed.rank: ",core_context.distributed.rank)
            if latest_checkpoint is not None:
                print("Loading Checkpoint")
                model_ckpt, epoch = load_state(latest_checkpoint, trial_id,core_context)
                trainer.start_epoch = epoch
            train(core_context,
                  trainer, 
                  core_context.distributed.size, 
                  batch_size, 
                  RANK,
                  trial_id)
    

if __name__ == '__main__':
    # args= parser.parse_args()
    info = det.get_cluster_info()
    print("Info")
    print(info)
    print("info.trial.trial_id: ",info.trial.trial_id)
    hparams = info.trial.hparams
    
    print("hparams: ",hparams)
    assert info is not None, "this example only runs on-cluster"
    latest_checkpoint = info.latest_checkpoint
    if latest_checkpoint is None:
        print("latest_checkpoint None: ",latest_checkpoint)
    else:
        print("latest_checkpoint: ",latest_checkpoint)
    '''
    5/19/23    Limitation - Need to know number of epochs before hand to do learning rate warmup
    '''
    nprocs = torch.cuda.device_count()
    local_rank=-1
    # 1a. check if we run in distributed training mode,
    # we suppose LOCAL_RANK is not defined if that's not the case
    if hparams['mult'] ==True and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        # args.local_rank=local_rank
        print("os.environ['LOCAL_RANK']", os.environ['LOCAL_RANK'])
    
    else:
        # Hack, if running normal loop, ignore torch's count of GPUs
        world_size=1
        local_rank=-1
    print("local_rank: ",local_rank)
    # else:
    #     world_size = nprocs
    print(local_rank,nprocs)
    main(local_rank,nprocs,hparams,latest_checkpoint,info.trial.trial_id)
# if __name__ == "__main__":
#         main(local_rank,world_size)