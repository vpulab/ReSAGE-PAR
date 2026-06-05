import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmcv.cnn import get_model_complexity_info
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

from configs import cfg, update_config
from dataset.multi_label.coco import COCO14
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_ema import ModelEmaV2
from optim.adamw import AdamW
from scheduler.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from scheduler.cosine_lr import CosineLRScheduler
from tools.distributed import distribute_bn
from tools.vis import tb_visualizer_pedes
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from dataset.pedes_attr.pedes import PedesAttr
from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool
from models.backbone import swin_transformer, resnet, bninception #, vit
#from models.backbone.tresnet import tresnet
from losses import bceloss, scaledbceloss, weightedlossbce
from models import base_block




# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)

attr_names=None

def main(cfg, args):
    global attr_names

    seed = cfg.SEED
    set_seed(seed)


    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)

    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, f'ckpt_max_{time_str()}.pth')

    visdom = None
    if cfg.VIS.VISDOM:
        visdom = Visdom(env=f'{cfg.DATASET.NAME}_' + cfg.NAME, port=8401)
        assert visdom.check_connection()

    writer = None
    if cfg.VIS.TENSORBOARD.ENABLE:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        writer_dir = os.path.join(exp_dir, cfg.NAME, 'runs', current_time)
        writer = SummaryWriter(log_dir=writer_dir)

    if cfg.REDIRECTOR:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    """
    the reason for args usage is CfgNode is immutable
    """
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = None

    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(f'use GPU{args.device} for training')
        print(args.world_size, args.rank)

    if args.local_rank == 0:
        print(cfg)

    train_tsfm, valid_tsfm = get_transform(cfg)
    if args.local_rank == 0:
        print(train_tsfm)

    if cfg.DATASET.TYPE == 'pedes':
        train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
                              target_transform=cfg.DATASET.TARGETTRANSFORM, train=True)

        valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                              target_transform=cfg.DATASET.TARGETTRANSFORM, train=False)
        attr_names = train_set.attr_id
    elif cfg.DATASET.TYPE == 'multi_label':
        train_set = COCO14(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
                           target_transform=cfg.DATASET.TARGETTRANSFORM)

        valid_set = COCO14(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                           target_transform=cfg.DATASET.TARGETTRANSFORM)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    print("train sample is None")
    print(train_sampler is None)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.local_rank == 0:
        print('-' * 60)
        print(f'{cfg.DATASET.NAME} attr_num : {train_set.attr_num}, eval_attr_num : {train_set.eval_attr_num} '
              f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_loader.dataset)}, '
              f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
              )

    labels = train_set.label
    label_ratio = labels.mean(0) if cfg.LOSS.SAMPLE_WEIGHT else None
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=train_set.attr_num,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)
    if args.local_rank == 0:
        print(f"backbone: {cfg.BACKBONE.TYPE}, classifier: {cfg.CLASSIFIER.NAME}")
        print(f"model_name: {cfg.NAME}")

    # flops, params = get_model_complexity_info(model, (3, 256, 128), print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    model = model.cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)

    model_ema = None
    if cfg.TRAIN.EMA.ENABLE:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=cfg.TRAIN.EMA.DECAY, device='cpu' if cfg.TRAIN.EMA.FORCE_CPU else None)

    if cfg.RELOAD.TYPE:
        model = get_reload_weight(model_dir, model, pth=cfg.RELOAD.PTH)

    loss_weight = cfg.LOSS.LOSS_WEIGHT

    if cfg.LOSS.TYPE == "weightedbceloss":
        pos_weight=train_set.pos_weigth
        pos_weight_tensor = torch.Tensor(pos_weight)
        pos_weight_tensor.to("cuda")
        criterion = build_loss(cfg.LOSS.TYPE)(
                sample_weight=label_ratio, scale=cfg.CLASSIFIER.SCALE, size_sum=cfg.LOSS.SIZESUM, tb_writer=writer, pos_weight_tensor=pos_weight_tensor)
        criterion = criterion.cuda()
    else:
        criterion = build_loss(cfg.LOSS.TYPE)(
            sample_weight=label_ratio, scale=cfg.CLASSIFIER.SCALE, size_sum=cfg.LOSS.SIZESUM, tb_writer=writer)
        criterion = criterion.cuda()

    if cfg.TRAIN.BN_WD:
        param_groups = [{'params': model.module.finetune_params(),
                         'lr': cfg.TRAIN.LR_SCHEDULER.LR_FT,
                         'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY},
                        {'params': model.module.fresh_params(),
                         'lr': cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                         'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}]
    else:
        # bn parameters are not applied with weight decay
        ft_params = seperate_weight_decay(
            model.module.finetune_params(),
            lr=cfg.TRAIN.LR_SCHEDULER.LR_FT,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

        fresh_params = seperate_weight_decay(
            model.module.fresh_params(),
            lr=cfg.TRAIN.LR_SCHEDULER.LR_NEW,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

        param_groups = ft_params + fresh_params

    if cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adamw':
        optimizer = AdamW(param_groups)
    else:
        assert None, f'{cfg.TRAIN.OPTIMIZER.TYPE} is not implemented'

    if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
        if cfg.CLASSIFIER.BN:
            assert False, 'BN can not compatible with ReduceLROnPlateau'
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_SCHEDULER.LR_STEP, gamma=0.1)
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine':
        lr_scheduler = CosineAnnealingLR_with_Restart(
            optimizer,
            T_max=(cfg.TRAIN.MAX_EPOCH + 5) * len(train_loader),
            T_mult=1,
            eta_min=cfg.TRAIN.LR_SCHEDULER.LR_NEW * 0.001
    )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':


        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.TRAIN.MAX_EPOCH,
            lr_min=1e-5,  # cosine lr 最终回落的位置
            warmup_lr_init=1e-4,
            warmup_t=cfg.TRAIN.MAX_EPOCH * cfg.TRAIN.LR_SCHEDULER.WMUP_COEF,
        )

    else:
        assert False, f'{cfg.LR_SCHEDULER.TYPE} has not been achieved yet'

    best_metric, epoch = trainer(cfg, args, epoch=cfg.TRAIN.MAX_EPOCH,
                                 model=model, model_ema=model_ema,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path,
                                 loss_w=loss_weight,
                                 viz=visdom,
                                 tb_writer=writer)
    if args.local_rank == 0:
        print(f'{cfg.NAME},  best_metrc : {best_metric} in epoch{epoch}')

import mlflow
def trainer(cfg, args, epoch, model, model_ema, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path, loss_w, viz, tb_writer):
    maximum = float(-np.inf)
    best_epoch = 0
    global attr_names
    #mlflow.pytorch.autolog()

    if cfg.SAVE.LR:
        model_saved = False
        first_lr = optimizer.param_groups[1]['lr']
        best_model_next_epoch = False

    result_list = defaultdict()

    result_path = path
    result_path = result_path.replace('ckpt_max', 'metric')
    result_path = result_path.replace('pth', 'pkl')

    descriptionForMLFlow = 'seed '+str(cfg.SEED)+' model '+cfg.BACKBONE.TYPE+' dataset '+cfg.DATASET.NAME+' zs: '+str(cfg.DATASET.ZERO_SHOT)+' input image dim: '+str(cfg.DATASET.HEIGHT)+' x '+str(cfg.DATASET.WIDTH)

    with mlflow.start_run(description=descriptionForMLFlow):
        params = {
                "epochs": epoch,
                "learning_rate type": cfg.TRAIN.LR_SCHEDULER.TYPE,
                "learning_rate ft": cfg.TRAIN.LR_SCHEDULER.LR_FT,
                "learning_rate new": cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                "batch_size": cfg.TRAIN.BATCH_SIZE,
                "loss_function": cfg.LOSS.TYPE,
                "optimizer": cfg.TRAIN.OPTIMIZER,
            }
        # Log training parameters.
        mlflow.log_params(params)

        for e in range(epoch):

            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            lr = optimizer.param_groups[1]['lr']

            mlflow.log_metric("learning_rate_"+str(cfg.SEED), f"{lr:.1e}", step=e)

            train_loss, train_gt, train_probs, train_imgs, train_logits, train_loss_mtr = batch_trainer(
                cfg,
                args=args,
                epoch=e,
                model=model,
                model_ema=model_ema,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                loss_w=loss_w,
                scheduler=lr_scheduler if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' else None,
            )

            if args.distributed:
                if args.local_rank == 0:
                    print("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if model_ema is not None and not cfg.TRAIN.EMA.FORCE_CPU:

                if args.local_rank == 0:
                    print('using model_ema to validate')

                if args.distributed:
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                valid_loss, valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_trainer(
                    cfg,
                    args=args,
                    epoch=e,
                    model=model_ema.module,
                    valid_loader=valid_loader,
                    criterion=criterion,
                    loss_w=loss_w
                )
            else:
                valid_loss, valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_trainer(
                    cfg,
                    args=args,
                    epoch=e,
                    model=model,
                    valid_loader=valid_loader,
                    criterion=criterion,
                    loss_w=loss_w
                )

            if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
                lr_scheduler.step(metrics=valid_loss)
            elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':
                lr_scheduler.step(epoch=e + 1)
            elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
                lr_scheduler.step()

            if cfg.METRIC.TYPE == 'pedestrian':
                #print(train_probs)
                train_result = get_pedestrian_metrics(train_gt, train_probs, index=None, cfg=cfg)
                valid_result = get_pedestrian_metrics(valid_gt, valid_probs, index=None, cfg=cfg)

                # aqui poner el mlflow para las estadisticas acc, ma
                # y el learning rate

                if args.local_rank == 0:
                    print(f'Evaluation on train set, train losses {train_loss}\n',
                        'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                            train_result.ma, np.mean(train_result.label_f1),
                            np.mean(train_result.label_pos_recall),
                            np.mean(train_result.label_neg_recall)),
                        'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                            train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                            train_result.instance_f1))

                    mlflow.log_metric("training_ma_"+str(cfg.SEED), f"{train_result.ma:3f}", step=e)
                    mlflow.log_metric("training_label_f1_"+str(cfg.SEED), f"{np.mean(train_result.label_f1):3f}", step=e)
                    mlflow.log_metric("training_pos_recall_"+str(cfg.SEED), f"{np.mean(train_result.label_pos_recall):3f}", step=e)
                    mlflow.log_metric("training_neg_recall_"+str(cfg.SEED), f"{np.mean(train_result.label_neg_recall):3f}", step=e)
                    mlflow.log_metric("training_acc_"+str(cfg.SEED), f"{train_result.instance_acc:3f}", step=e)
                    mlflow.log_metric("training_prec_"+str(cfg.SEED), f"{train_result.instance_prec:3f}", step=e)
                    mlflow.log_metric("training_rec_"+str(cfg.SEED), f"{train_result.instance_recall:3f}", step=e)
                    mlflow.log_metric("training_f1_"+str(cfg.SEED), f"{train_result.instance_f1:3f}", step=e)

                    for attribute in attr_names:
                        if '&' in attribute:
                            strAttribute = attribute.replace('&', '')
                            mlflow.log_metric("training_"+strAttribute+"_ma_"+str(cfg.SEED), f"{train_result.label_ma[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+strAttribute+"_label_f1_"+str(cfg.SEED), f"{train_result.label_f1[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+strAttribute+"_label_pos_recall_"+str(cfg.SEED), f"{train_result.label_pos_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+strAttribute+"_label_neg_recall_"+str(cfg.SEED), f"{train_result.label_neg_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+strAttribute+"_label_acc_"+str(cfg.SEED), f"{train_result.label_acc[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+strAttribute+"_label_prec_"+str(cfg.SEED), f"{train_result.label_prec[attr_names.index(attribute)]:3f}", step=e)

                        else:
                            mlflow.log_metric("training_"+attribute+"_ma_"+str(cfg.SEED), f"{train_result.label_ma[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+attribute+"_label_f1_"+str(cfg.SEED), f"{train_result.label_f1[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+attribute+"_label_pos_recall_"+str(cfg.SEED), f"{train_result.label_pos_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+attribute+"_label_neg_recall_"+str(cfg.SEED), f"{train_result.label_neg_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+attribute+"_label_acc_"+str(cfg.SEED), f"{train_result.label_acc[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("training_"+attribute+"_label_prec_"+str(cfg.SEED), f"{train_result.label_prec[attr_names.index(attribute)]:3f}", step=e)


                    print(f'Evaluation on test set, valid losses {valid_loss}\n',
                        'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                            valid_result.ma, np.mean(valid_result.label_f1),
                            np.mean(valid_result.label_pos_recall),
                            np.mean(valid_result.label_neg_recall)),
                        'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                            valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                            valid_result.instance_f1))

                    mlflow.log_metric("testing_ma_"+str(cfg.SEED), f"{valid_result.ma:3f}", step=e)
                    mlflow.log_metric("testing_label_f1_"+str(cfg.SEED), f"{np.mean(valid_result.label_f1):3f}", step=e)
                    mlflow.log_metric("testing_pos_recall_"+str(cfg.SEED), f"{np.mean(valid_result.label_pos_recall):3f}", step=e)
                    mlflow.log_metric("testing_neg_recall_"+str(cfg.SEED), f"{np.mean(valid_result.label_neg_recall):3f}", step=e)
                    mlflow.log_metric("testing_acc_"+str(cfg.SEED), f"{valid_result.instance_acc:3f}", step=e)
                    mlflow.log_metric("testing_prec_"+str(cfg.SEED), f"{valid_result.instance_prec:3f}", step=e)
                    mlflow.log_metric("testing_rec_"+str(cfg.SEED), f"{valid_result.instance_recall:3f}", step=e)
                    mlflow.log_metric("testing_f1_"+str(cfg.SEED), f"{valid_result.instance_f1:3f}", step=e)

                    for attribute in attr_names:
                        if '&' in attribute:
                            strAttribute = attribute.replace('&', '')
                            mlflow.log_metric("testing_"+strAttribute+"_ma_"+str(cfg.SEED), f"{valid_result.label_ma[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+strAttribute+"_label_f1_"+str(cfg.SEED), f"{valid_result.label_f1[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+strAttribute+"_label_pos_recall_"+str(cfg.SEED), f"{valid_result.label_pos_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+strAttribute+"_label_neg_recall_"+str(cfg.SEED), f"{valid_result.label_neg_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+strAttribute+"_label_acc_"+str(cfg.SEED), f"{valid_result.label_acc[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+strAttribute+"_label_prec_"+str(cfg.SEED), f"{valid_result.label_prec[attr_names.index(attribute)]:3f}", step=e)
                         
                        else:
                            mlflow.log_metric("testing_"+attribute+"_ma_"+str(cfg.SEED), f"{valid_result.label_ma[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+attribute+"_label_f1_"+str(cfg.SEED), f"{valid_result.label_f1[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+attribute+"_label_pos_recall_"+str(cfg.SEED), f"{valid_result.label_pos_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+attribute+"_label_neg_recall_"+str(cfg.SEED), f"{valid_result.label_neg_recall[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+attribute+"_label_acc_"+str(cfg.SEED), f"{valid_result.label_acc[attr_names.index(attribute)]:3f}", step=e)
                            mlflow.log_metric("testing_"+attribute+"_label_prec_"+str(cfg.SEED), f"{valid_result.label_prec[attr_names.index(attribute)]:3f}", step=e)

                    saveEpochMetrics(train_result, valid_result, attr_names, lr)
     
                    print(f'{time_str()}')
                    print('-' * 60)

                if args.local_rank == 0:
                    tb_visualizer_pedes(tb_writer, lr, e, train_loss, valid_loss, train_result, valid_result,
                                        train_gt, valid_gt, train_loss_mtr, valid_loss_mtr, model, train_loader.dataset.attr_id)


                if cfg.SAVE.LR:
                    if best_model_next_epoch == True and model_saved == False:
                        maximum = valid_result.ma
                        best_epoch = e
                        save_ckpt(model, path, e, maximum)
                        saveMLFLOWBestMetrics(mlflow, train_result, valid_result, train_imgs, valid_imgs, train_probs, valid_probs, train_gt, valid_gt)
                        model_saved = True
                    else:
                        actual_lr = optimizer.param_groups[1]['lr']
                        if actual_lr < first_lr:
                            # wait one epoch to get the best model
                            best_model_next_epoch = True

                else:
                    cur_metric = valid_result.ma
                    if cur_metric > maximum:
                        maximum = cur_metric
                        best_epoch = e
                        save_ckpt(model, path, e, maximum)
                        saveMLFLOWBestMetrics(mlflow, train_result, valid_result, train_imgs, valid_imgs, train_probs, valid_probs, train_gt, valid_gt)



                result_list[e] = {
                    'train_result': train_result,  # 'train_map': train_map,
                    'valid_result': valid_result,  # 'valid_map': valid_map,
                    'train_gt': train_gt, 'train_probs': train_probs,
                    'valid_gt': valid_gt, 'valid_probs': valid_probs,
                    'train_imgs': train_imgs, 'valid_imgs': valid_imgs
                }

            elif cfg.METRIC.TYPE == 'multi_label':

                train_metric = get_multilabel_metrics(train_gt, train_probs)
                valid_metric = get_multilabel_metrics(valid_gt, valid_probs)

                if args.local_rank == 0:
                    print(
                        'Train Performance : mAP: {:.4f}, OP: {:.4f}, OR: {:.4f}, OF1: {:.4f} CP: {:.4f}, CR: {:.4f}, '
                        'CF1: {:.4f}'.format(train_metric.map, train_metric.OP, train_metric.OR, train_metric.OF1,
                                            train_metric.CP, train_metric.CR, train_metric.CF1))

                    print(
                        'Test Performance : mAP: {:.4f}, OP: {:.4f}, OR: {:.4f}, OF1: {:.4f} CP: {:.4f}, CR: {:.4f}, '
                        'CF1: {:.4f}'.format(valid_metric.map, valid_metric.OP, valid_metric.OR, valid_metric.OF1,
                                            valid_metric.CP, valid_metric.CR, valid_metric.CF1))
                    print(f'{time_str()}')
                    print('-' * 60)

                    tb_writer.add_scalars('train/lr', {'lr': lr}, e)

                    tb_writer.add_scalars('train/losses', {'train': train_loss,
                                                        'test': valid_loss}, e)

                    tb_writer.add_scalars('train/perf', {'mAP': train_metric.map,
                                                        'OP': train_metric.OP,
                                                        'OR': train_metric.OR,
                                                        'OF1': train_metric.OF1,
                                                        'CP': train_metric.CP,
                                                        'CR': train_metric.CR,
                                                        'CF1': train_metric.CF1}, e)

                    tb_writer.add_scalars('test/perf', {'mAP': valid_metric.map,
                                                        'OP': valid_metric.OP,
                                                        'OR': valid_metric.OR,
                                                        'OF1': valid_metric.OF1,
                                                        'CP': valid_metric.CP,
                                                        'CR': valid_metric.CR,
                                                        'CF1': valid_metric.CF1}, e)

                cur_metric = valid_metric.map
                if cur_metric > maximum:
                    maximum = cur_metric
                    best_epoch = e
                    save_ckpt(model, path, e, maximum)

                result_list[e] = {
                    'train_result': train_metric, 'valid_result': valid_metric,
                    'train_gt': train_gt, 'train_probs': train_probs,
                    'valid_gt': valid_gt, 'valid_probs': valid_probs
                }
            else:
                assert False, f'{cfg.METRIC.TYPE} is unavailable'

            with open(result_path, 'wb') as f:
                pickle.dump(result_list, f)
        
        saveMLFLOWMetrics(mlflow)

    return maximum, best_epoch

training_dict_metrics_mlflow = {}
testing_dict_metrics_mlflow = {}

training_dict_label_metrics_mlflow = {}
testing_dict_label_metrics_mlflow = {}

def saveMLFLOWMetrics(mlflow):
    global training_dict_metrics_mlflow
    global testing_dict_metrics_mlflow
    global training_dict_label_metrics_mlflow
    global testing_dict_label_metrics_mlflow

    mlflow.log_dict(training_dict_metrics_mlflow, "training_metrics.json")
    mlflow.log_dict(testing_dict_metrics_mlflow, "testing_metrics.json")
    
    mlflow.log_dict(training_dict_label_metrics_mlflow,"training_metrics_labels.json")
    mlflow.log_dict(testing_dict_label_metrics_mlflow, "testing_metrics_labels.json")  



def saveEpochMetrics(train_result, valid_result, attr_names, lr):
    metrics = ["ma", "prec", "acc", "rec", "f1", "pos recall", "neg recall"]  
    global training_dict_metrics_mlflow
    global testing_dict_metrics_mlflow
    global training_dict_label_metrics_mlflow
    global testing_dict_label_metrics_mlflow

    if not training_dict_metrics_mlflow:
        training_dict_metrics_mlflow["ma"] = [train_result.ma.tolist()]
        training_dict_metrics_mlflow["prec"] = [train_result.instance_prec.tolist()]
        training_dict_metrics_mlflow["acc"] = [train_result.instance_acc.tolist()]
        training_dict_metrics_mlflow["rec"] = [train_result.instance_recall.tolist()]
        training_dict_metrics_mlflow["f1"] = [train_result.instance_f1.tolist()]
        training_dict_metrics_mlflow["lr"] = [lr]
    else:
        training_dict_metrics_mlflow["ma"].append(train_result.ma.tolist())
        training_dict_metrics_mlflow["prec"].append(train_result.instance_prec.tolist())
        training_dict_metrics_mlflow["acc"].append(train_result.instance_acc.tolist())
        training_dict_metrics_mlflow["rec"].append(train_result.instance_recall.tolist())
        training_dict_metrics_mlflow["f1"].append(train_result.instance_f1.tolist())
        training_dict_metrics_mlflow["lr"].append(lr)

    if not testing_dict_metrics_mlflow:
        testing_dict_metrics_mlflow["ma"] = [valid_result.ma.tolist()]
        testing_dict_metrics_mlflow["prec"] = [valid_result.instance_prec.tolist()]
        testing_dict_metrics_mlflow["acc"] = [valid_result.instance_acc.tolist()]
        testing_dict_metrics_mlflow["rec"] = [valid_result.instance_recall.tolist()]
        testing_dict_metrics_mlflow["f1"] = [valid_result.instance_f1.tolist()]
        testing_dict_metrics_mlflow["lr"] = [lr]
    else:
        testing_dict_metrics_mlflow["ma"].append(valid_result.ma.tolist())
        testing_dict_metrics_mlflow["prec"].append(valid_result.instance_prec.tolist())
        testing_dict_metrics_mlflow["acc"].append(valid_result.instance_acc.tolist())
        testing_dict_metrics_mlflow["rec"].append(valid_result.instance_recall.tolist())
        testing_dict_metrics_mlflow["f1"].append(valid_result.instance_f1.tolist())
        testing_dict_metrics_mlflow["lr"].append(lr)
    
    # labels metrics
    for attribute in attr_names:
        # training
        if "ma_"+attribute not in training_dict_label_metrics_mlflow.keys():
            training_dict_label_metrics_mlflow["ma_"+attribute] = [train_result.label_ma[attr_names.index(attribute)].tolist()]
            training_dict_label_metrics_mlflow["prec_"+attribute] = [train_result.label_prec[attr_names.index(attribute)].tolist()]
            training_dict_label_metrics_mlflow["acc_"+attribute] = [train_result.label_acc[attr_names.index(attribute)].tolist()]
            training_dict_label_metrics_mlflow["posrec_"+attribute] = [train_result.label_pos_recall[attr_names.index(attribute)].tolist()]
            training_dict_label_metrics_mlflow["negrec_"+attribute] = [train_result.label_neg_recall[attr_names.index(attribute)].tolist()]
            training_dict_label_metrics_mlflow["f1_"+attribute] = [train_result.label_f1[attr_names.index(attribute)].tolist()]
        else:
            training_dict_label_metrics_mlflow["ma_"+attribute].append(train_result.label_ma[attr_names.index(attribute)].tolist())
            training_dict_label_metrics_mlflow["prec_"+attribute].append(train_result.label_prec[attr_names.index(attribute)].tolist())
            training_dict_label_metrics_mlflow["acc_"+attribute].append(train_result.label_acc[attr_names.index(attribute)].tolist())
            training_dict_label_metrics_mlflow["posrec_"+attribute].append(train_result.label_pos_recall[attr_names.index(attribute)].tolist())
            training_dict_label_metrics_mlflow["negrec_"+attribute].append(train_result.label_neg_recall[attr_names.index(attribute)].tolist())
            training_dict_label_metrics_mlflow["f1_"+attribute].append(train_result.label_f1[attr_names.index(attribute)].tolist())

        # testing
        if "ma_"+attribute not in testing_dict_label_metrics_mlflow.keys():
            testing_dict_label_metrics_mlflow["ma_"+attribute] = [valid_result.label_ma[attr_names.index(attribute)].tolist()]
            testing_dict_label_metrics_mlflow["prec_"+attribute] = [valid_result.label_prec[attr_names.index(attribute)].tolist()]
            testing_dict_label_metrics_mlflow["acc_"+attribute] = [valid_result.label_acc[attr_names.index(attribute)].tolist()]
            testing_dict_label_metrics_mlflow["posrec_"+attribute] = [valid_result.label_pos_recall[attr_names.index(attribute)].tolist()]
            testing_dict_label_metrics_mlflow["negrec_"+attribute] = [valid_result.label_neg_recall[attr_names.index(attribute)].tolist()]
            testing_dict_label_metrics_mlflow["f1_"+attribute] = [valid_result.label_f1[attr_names.index(attribute)].tolist()]
        else:
            testing_dict_label_metrics_mlflow["ma_"+attribute].append(valid_result.label_ma[attr_names.index(attribute)].tolist())
            testing_dict_label_metrics_mlflow["prec_"+attribute].append(valid_result.label_prec[attr_names.index(attribute)].tolist())
            testing_dict_label_metrics_mlflow["acc_"+attribute].append(valid_result.label_acc[attr_names.index(attribute)].tolist())
            testing_dict_label_metrics_mlflow["posrec_"+attribute].append(valid_result.label_pos_recall[attr_names.index(attribute)].tolist())
            testing_dict_label_metrics_mlflow["negrec_"+attribute].append(valid_result.label_neg_recall[attr_names.index(attribute)].tolist())
            testing_dict_label_metrics_mlflow["f1_"+attribute].append(valid_result.label_f1[attr_names.index(attribute)].tolist())
    


def saveMLFLOWBestMetrics(mlflow, train_result, valid_result, train_imgs, valid_imgs, train_probs, valid_probs, train_gt, valid_gt):

    # save instance metrics
    # f1 metric training and testing
    training_dict_f1 = {
        "training f1": train_result.instance_f1_label.tolist(),
        "img name training": train_imgs,
    }
    mlflow.log_dict(training_dict_f1, "training_dict_instance_f1.json")

    testing_dict_f1 = {
        "testing f1": valid_result.instance_f1_label.tolist(),
        "img name testing": valid_imgs,
    }
    mlflow.log_dict(testing_dict_f1, "testing_dict_instance_f1.json")

    # acc metric training and testing
    training_dict_acc = {
        "training acc": train_result.instance_acc_label.tolist(),
        "img name training": train_imgs,
    }
    mlflow.log_dict(training_dict_acc, "training_dict_instance_acc.json")

    testing_dict_acc = {
        "testing acc": valid_result.instance_acc_label.tolist(),
        "img name testing": valid_imgs,
    }
    mlflow.log_dict(testing_dict_acc, "testing_dict_instance_acc.json")

    # rec metric training and testing
    training_dict_rec = {
        "training rec": train_result.instance_recall_label.tolist(),
        "img name training": train_imgs,
    }
    mlflow.log_dict(training_dict_rec, "training_dict_instance_rec.json")

    testing_dict_rec = {
        "testing rec": valid_result.instance_recall_label.tolist(),
        "img name testing": valid_imgs,
    }
    mlflow.log_dict(testing_dict_rec, "testing_dict_instance_rec.json")

    # ma metric training and testing
    training_dict_ma = {
        "training ma": train_result.ma.tolist(),
        "img name training": train_imgs,
    }
    mlflow.log_dict(training_dict_ma, "training_dict_instance_ma.json")

    testing_dict_ma = {
        "testing ma": valid_result.ma.tolist(),
        "img name testing": valid_imgs,
    }
    mlflow.log_dict(testing_dict_ma, "testing_dict_instance_ma.json")

    threshold = 0.5

    training_prob_dict = {
        "training gt": train_gt.tolist(),
        "training prob predicted": train_probs.tolist(),
    }
    
    mlflow.log_dict(training_prob_dict, "training_dict_gt_prob_predicted.json")

    testing_prob_dict = {
        "testing gt": valid_gt.tolist(),
        "testing prob predicted": valid_probs.tolist(),
    }

    mlflow.log_dict(testing_prob_dict, "testing_dict_gt_prob_predicted.json")


    pred_training = train_probs > threshold
    pred_testing = valid_probs > threshold

    training_dict = {
        "training gt": train_gt.tolist(),
        "training predicted": pred_training.tolist(),
    }
    
    mlflow.log_dict(training_dict, "training_dict_gt_predicted.json")

    testing_dict = {
        "testing gt": valid_gt.tolist(),
        "testing predicted": pred_testing.tolist(),
    }

    mlflow.log_dict(testing_dict, "testing_dict_gt_predicted.json")

    return


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/pedes_baseline/pa100k.yaml",

    )

    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    parser.add_argument('--dist_bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    update_config(cfg, args)
    main(cfg, args)
