'''
Training script for video object detection
'''
import os
import argparse
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

from models.general import name_to_model
from utils import timer, image_ops, optim
from datasets import get_trainingset
from utils.evaluation import get_valset
import api
from settings import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     type=str, default='yv3a1_agg_dev')
    parser.add_argument('--train_set', type=str, default='HBMWR_mot_train')
    parser.add_argument('--val_set',   type=str, default='Lab1_mot')

    parser.add_argument('--super_batchsize', type=int,   default=32)
    parser.add_argument('--initial_imgsize', type=int,   default=None)
    parser.add_argument('--optimizer',       type=str,   default='SGDMR')
    parser.add_argument('--optim_params',    type=str,   default='all')
    parser.add_argument('--lr',              type=float, default=0.0001)
    parser.add_argument('--warmup',          type=int,   default=1000)
    parser.add_argument('--checkpoint',      type=str,
                        default='rapid_H1MW1024_Mar11_4000_.pth')

    parser.add_argument('--print_interval',      type=int, default=20)
    parser.add_argument('--eval_interval',       type=int, default=200)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    parser.add_argument('--demo_interval',       type=int, default=100)
    parser.add_argument('--demo_images',         type=str, default='fisheye')
    
    parser.add_argument('--debug_mode', type=str, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    print('Initialing model...')
    model, global_cfg = name_to_model(args.model)

    # -------------------------- settings ---------------------------
    if args.debug_mode == 'overfit':
        raise NotImplementedError()
        print(f'Running debug mode: {args.debug_mode}...')
        # overfitting on one or a few images
        global_cfg['train.img_sizes'] = [640]
        global_cfg['train.initial_imgsize'] = 640
        global_cfg['test.preprocessing'] = 'resize_pad_square'
        target_size = 640
        global_cfg['train.data_augmentation'] = None
        enable_multiscale = False
        batch_size = 1
        subdivision = 1
        num_cpu = 0
        warmup_iter = 40
    elif args.debug_mode == 'local':
        print(f'Running debug mode: {args.debug_mode}...')
        # train on local laptop with a small resolution and batch size
        TRAIN_RESOLUTIONS = [384, 512]
        AUTO_BATCHSIZE = {'384': 4, '512': 2}
        initial_size = TRAIN_RESOLUTIONS[-1]
        global_cfg['train.initial_imgsize'] = initial_size
        batch_size = 2
        seq_len = global_cfg['train.sequence_length']
        super_batchsize = args.super_batchsize
        subdivision = int(np.ceil(super_batchsize / batch_size / seq_len))
        # data augmentation setting
        enable_multiscale = True
        num_cpu = 0
        warmup_iter = args.warmup
        # testing setting
        target_size = global_cfg.get('test.default_input_size', None)
    elif args.debug_mode == None:
        print(f'Debug mode disabled.')
        # normal training
        AUTO_BATCHSIZE = global_cfg['train.imgsize_to_batch_size']
        TRAIN_RESOLUTIONS = global_cfg['train.img_sizes']
        if args.initial_imgsize is not None:
            initial_size = args.initial_imgsize
            assert initial_size in TRAIN_RESOLUTIONS
        else:
            initial_size = TRAIN_RESOLUTIONS[-1]
        global_cfg['train.initial_imgsize'] = initial_size
        batch_size = AUTO_BATCHSIZE[str(initial_size)]
        seq_len = global_cfg['train.sequence_length']
        super_batchsize = args.super_batchsize
        subdivision = int(np.ceil(super_batchsize / batch_size / seq_len))
        # data augmentation setting
        enable_multiscale = True
        assert 'train.imgsize_to_batch_size' in global_cfg
        print('Auto-batchsize enabled. Automatically selecting the batch size.')
        num_cpu = 4
        warmup_iter = args.warmup
        # testing setting
        target_size = global_cfg.get('test.default_input_size', None)
    else:
        raise Exception('Unknown debug mode')

    job_name = f'{args.model}_{args.train_set}_{args.lr}'
    
    # Prepare model
    pnum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters of {args.model} =', pnum)
    model = model.cuda()
    model.train()

    # Training set and validation set setting
    print(f'Initializing training set {args.train_set}...')
    global_cfg['train.dataset_name'] = args.train_set
    dataset = get_trainingset(global_cfg)
    dataset.to_iterator(batch_size=batch_size, num_workers=num_cpu, pin_memory=True)
    print(f'Initializing validation set {args.val_set}...')
    eval_info, validation_func = get_valset(args.val_set)

    start_iter = -1
    if args.checkpoint:
        print("Loading checkpoint...", args.checkpoint)
        weights_path = os.path.join(f'{PROJECT_ROOT}/weights', args.checkpoint)
        previous_state = torch.load(weights_path)
        if 'input' in global_cfg['model.agg.hidden_state_names']:
            for k in list(previous_state['model'].keys()):
                if 'netlist.0' in k:
                    previous_state['model'].pop(k)
        try:
            model.load_state_dict(previous_state['model'])
        except:
            print('Cannot load weights. Trying to set strict=False...')
            model.load_state_dict(previous_state['model'], strict=False)
            print('Successfully loaded part of the weights.')
        start_iter = previous_state.get('iter', start_iter)
        print(f'Start from iteration: {start_iter}')

    print('Initializing tensorboard SummaryWriter...')
    if args.debug_mode:
        logger = SummaryWriter(f'{PROJECT_ROOT}logs/debug/{job_name}')
    else:
        logger = SummaryWriter(f'{PROJECT_ROOT}logs/{job_name}')

    print(f'Initializing optimizer with lr: {args.lr}')
    # set weight decay only on conv.weight
    params = []
    if args.optim_params == 'all':
        for key, value in model.named_parameters():
            decay = global_cfg['train.sgd.weight_decay'] if 'conv' in key else 0.0
            params += [{'params': value, 'weight_decay': decay}]
    elif args.optim_params == 'fix_backbone':
        for key, value in model.fpn.named_parameters():
            decay = global_cfg['train.sgd.weight_decay'] if 'conv' in key else 0.0
            params += [{'params': value, 'weight_decay': decay}]
        for key, value in model.agg.named_parameters():
            decay = global_cfg['train.sgd.weight_decay'] if 'conv' in key else 0.0
            params += [{'params': value, 'weight_decay': decay}]
        for key, value in model.rpn.named_parameters():
            decay = global_cfg['train.sgd.weight_decay'] if 'conv' in key else 0.0
            params += [{'params': value, 'weight_decay': decay}]
    elif args.optim_params == 'agg_only':
        for key, value in model.agg.named_parameters():
            decay = global_cfg['train.sgd.weight_decay'] if 'conv' in key else 0.0
            params += [{'params': value, 'weight_decay': decay}]
    else:
        raise NotImplementedError()
    pnum = sum(p['params'].numel() for p in params if p['params'].requires_grad)
    print(f'Number of training parameters =', pnum)
    # Initialize optimizer
    optimizer = optim.get_optimizer(name=args.optimizer, params=params,
                                    lr=args.lr, cfg=global_cfg)
    if args.checkpoint and args.optimizer in previous_state:
        try:
            optimizer.load_state_dict(previous_state[args.optimizer])
        except:
            print('Failed loading optimizer state. Initialize optimizer from scratch.')
            start_iter = -1
    # Learning rate scheduler
    lr_schedule_func = lambda x: lr_warmup(x, warm_up=warmup_iter)
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_schedule_func, last_epoch=start_iter)

    print('Start training...')
    today = timer.today()
    start_time = timer.tic()
    for iter_i in range(start_iter, 1000000):
        # evaluation
        if iter_i > 0 and iter_i % args.eval_interval == 0:
        # if iter_i % args.eval_interval == 0:
            if args.debug_mode != 'overfit':
                model.eval()
            model.clear_hidden_state()
            with timer.contexttimer() as t0:
                model_eval = api.Detector(model_and_cfg=(model, global_cfg))
                dts = model_eval.eval_predict_vod(eval_info,
                    input_size=target_size,
                    conf_thres=global_cfg.get('test.ap_conf_thres', 0.005))
                eval_str, ap, ap50, ap75 = validation_func(dts)
            del model_eval
            s = f'\nCurrent time: [ {timer.now()} ], iteration: [ {iter_i} ]\n\n'
            s += eval_str + '\n\n'
            s += f'Validation elapsed time: [ {t0.time_str} ]'
            print(s)
            logger.add_text('Validation summary', s, iter_i)
            logger.add_scalar('Validation AP[IoU=0.5]', ap50, iter_i)
            logger.add_scalar('Validation AP[IoU=0.75]', ap75, iter_i)
            logger.add_scalar('Validation AP[IoU=0.5:0.95]', ap, iter_i)
            model.train()

        torch.cuda.reset_max_memory_allocated(0)
        seq_len = dataset.seq_len
        # subdivision loop
        optimizer.zero_grad()
        for _ in range(subdivision):
            seq_imgs, seq_labels, seq_flags, img_ids = dataset.get_next()
            assert len(seq_imgs) == len(seq_labels) == len(seq_flags)
            # visualize the clip for debugging
            if False:
                for b in range(batch_size):
                    for _im, _lab in zip(seq_imgs, seq_labels):
                        _im = image_ops.img_tensor_to_np(_im[b],
                                    model.input_format, 'BGR_uint8')
                        _lab[b].draw_on_np(_im)
                        cv2.imshow('', _im)
                        cv2.waitKey(500)
            model.clear_hidden_state()
            for imgs, labels, is_start in zip(seq_imgs, seq_labels, seq_flags):
                imgs = imgs.cuda()
                loss = model(imgs, is_start, labels)
                assert not torch.isnan(loss)
                loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0/subdivision/seq_len)
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % args.print_interval == 0:
            sec_used  = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            _ai       = sec_used / (iter_i+1-start_iter)
            avg_iter  = timer.sec2str(_ai)
            avg_img = _ai / batch_size / subdivision / seq_len
            avg_100img   = timer.sec2str(avg_img * 100)
            avg_epoch = timer.sec2str(avg_img * 118287)
            print(f'\nTotal time: {time_used}, 100 imgs: {avg_100img}, ',
                  f'iter: {avg_iter}, COCO epoch: {avg_epoch}')
            print(f'effective batch size = {batch_size} * {subdivision} * {seq_len}')
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print(f'Max GPU memory usage: {max_cuda:.3f} GB')
            current_lr = scheduler.get_last_lr()[0]
            print(f'[Iteration {iter_i}] [learning rate {current_lr:.3g}]',
                  f'[Total loss {loss:.2f}] [img size {dataset.img_size}]')
            print(model.loss_str)

        # random resizing
        if enable_multiscale and iter_i > 0 and (iter_i % 10 == 0):
            # # Randomly pick a input resolution
            imgsize = np.random.choice(TRAIN_RESOLUTIONS)
            # Set the image size in datasets
            batch_size = AUTO_BATCHSIZE[str(imgsize)]
            subdivision = int(np.ceil(super_batchsize / batch_size / seq_len))
            dataset.img_size = imgsize
            dataset.to_iterator(batch_size=batch_size, num_workers=num_cpu,
                                pin_memory=True)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                args.optimizer: optimizer.state_dict(),
            }
            save_path = f'{PROJECT_ROOT}/weights/{job_name}_{today}_{iter_i}.pth'
            torch.save(state_dict, save_path)

        # save detection
        if iter_i > 0 and iter_i % args.demo_interval == 0:
            if args.debug_mode != 'overfit':
                model.eval()
            model_eval = api.Detector(model_and_cfg=(model, global_cfg))
            demo_images_dir = f'{PROJECT_ROOT}/images/{args.demo_images}'
            for imname in os.listdir():
                if not imname.endswith('.jpg'): continue
                impath = os.path.join(demo_images_dir, imname)
                model_eval.model.clear_hidden_state()
                np_img = model_eval.detect_one(img_path=impath, return_img=True,
                                        conf_thres=0.3, input_size=target_size)
                if args.debug_mode is not None:
                    cv2_im = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    log_dir = f'{PROJECT_ROOT}/logs/{args.model}_debug/'
                    if not os.path.exists(log_dir): os.mkdir(log_dir)
                    s = os.path.join(log_dir, f'{imname[:-4]}_iter{iter_i}.jpg')
                    cv2.imwrite(s, cv2_im)
                else:
                    if min(np_img.shape[:2]) > 512:
                        _h, _w = np_img.shape[:2]
                        _r = 512 / min(_h, _w)
                        np_img = cv2.resize(np_img, (int(_w*_r), int(_h*_r)))
                    logger.add_image(impath, np_img, iter_i, dataformats='HWC')
            model.train()


# Learning rate setup
def lr_warmup(i, warm_up=1000):
    if i < warm_up:
        factor = i / warm_up
    else:
        factor = 1
    # elif i < 70000:
    #     factor = 0.5
    return factor


if __name__ == "__main__":
    main()
