# This is the main training file we are using
import os
import argparse
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

from models.general import name_to_model
from datasets import get_trainingset
from utils.evaluation import get_valset
from utils import timer, image_ops, optim
from utils.structures import ImageObjects
import api
from settings import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     type=str, default='yolov3')
    parser.add_argument('--train_set', type=str, default='VID30train')
    parser.add_argument('--val_set',   type=str, default='VIDval2017new_100')

    parser.add_argument('--super_batchsize', type=int,   default=32)
    parser.add_argument('--initial_imgsize', type=int,   default=None)
    parser.add_argument('--optimizer',       type=str,   default='SGDMR')
    parser.add_argument('--lr',              type=float, default=0.0001)
    parser.add_argument('--warmup',          type=int,   default=1000)

    parser.add_argument('--checkpoint', type=str,
                        default='')

    parser.add_argument('--print_interval',      type=int, default=20)
    parser.add_argument('--eval_interval',       type=int, default=200)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    parser.add_argument('--demo_interval',       type=int, default=20)
    parser.add_argument('--demo_images',         type=str, default='VIDval2017new_100')
    
    parser.add_argument('--debug_mode',          type=str, default=None)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    print('Initialing model...')
    model, global_cfg = name_to_model(args.model)

    # -------------------------- settings ---------------------------
    ap_conf_thres = global_cfg.get('test.ap_conf_thres', 0.005)
    if args.debug_mode == 'overfit':
        print(f'Running debug mode: {args.debug_mode}...')
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
        super_batchsize = 8
        subdivision = int(np.ceil(super_batchsize / batch_size))
        # data augmentation setting
        enable_multiscale = True
        num_cpu = 0
        warmup_iter = args.warmup
        # testing setting
        target_size = global_cfg.get('test.default_input_size', None)
    elif args.debug_mode == None:
        # training setting
        TRAIN_RESOLUTIONS = global_cfg['train.img_sizes']
        AUTO_BATCHSIZE = global_cfg['train.imgsize_to_batch_size']
        if args.initial_imgsize is not None:
            initial_size = args.initial_imgsize
            assert initial_size in TRAIN_RESOLUTIONS
        else:
            initial_size = TRAIN_RESOLUTIONS[-1]
        global_cfg['train.initial_imgsize'] = initial_size
        batch_size = AUTO_BATCHSIZE[str(initial_size)]
        super_batchsize = args.super_batchsize
        subdivision = int(np.ceil(super_batchsize / batch_size))
        # data augmentation setting
        enable_multiscale = True
        assert 'train.imgsize_to_batch_size' in global_cfg
        print('Auto-batchsize enabled. Automatically selecting the batch size.')
        # optimizer setting
        num_cpu = 4 if global_cfg['train.hard_example_mining'] != 'probability' else 0
        warmup_iter = args.warmup
        # testing setting
        target_size = global_cfg.get('test.default_input_size', None)
    else:
        raise Exception('Unknown debug mode')

    job_name = f'{args.model}_{args.train_set}_{args.lr}'
    
    # Prepare model
    pnum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters in {args.model}:', pnum)
    model = model.cuda()
    model.train()

    # Training set and validation set setting
    print(f'Initializing training set {args.train_set}...')
    global_cfg['train.dataset_name'] = args.train_set
    dataset = get_trainingset(global_cfg)
    dataset.to_iterator(batch_size=batch_size, shuffle=True,
                        num_workers=num_cpu, pin_memory=True)
    print(f'Initializing validation set {args.val_set}...')
    eval_info, validation_func = get_valset(args.val_set)
    # validation function for hard example mining
    eval_func_ = eval_info['val_func']

    start_iter = -1
    if args.checkpoint:
        print("Loading checkpoint...", args.checkpoint)
        weights_path = f'{PROJECT_ROOT}/weights/{args.checkpoint}'
        previous_state = torch.load(weights_path)
        try:
            model.load_state_dict(previous_state['model'])
        except:
            print('Cannot load weights. Trying to set strict=False...')
            model.load_state_dict(previous_state['model'], strict=False)
            print('Successfully loaded part of the weights.')
        start_iter = previous_state.get('iter', -2) + 1
        print(f'Start from iteration: {start_iter}')

    print('Initializing tensorboard SummaryWriter...')
    if args.debug_mode:
        logger = SummaryWriter(f'{PROJECT_ROOT}/logs/debug/{job_name}')
    else:
        logger = SummaryWriter(f'{PROJECT_ROOT}/logs/{job_name}')

    print(f'Initializing optimizer with lr: {args.lr}')
    # set weight decay only on conv.weight
    params = []
    for key, value in model.named_parameters():
        decay = global_cfg['train.sgd.weight_decay'] if 'conv' in key else 0.0
        params += [{'params': value, 'weight_decay': decay}]
    # Initialize optimizer
    optimizer = optim.get_optimizer(name=args.optimizer, params=params,
                                    lr=args.lr, cfg=global_cfg)
    if args.checkpoint and args.optimizer in previous_state:
        optimizer.load_state_dict(previous_state[args.optimizer])
    # Learning rate scheduler
    lr_schedule_func = lambda x: optim.lr_warmup(x, warm_up=warmup_iter)
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_schedule_func, last_epoch=start_iter)

    print('Start training...')
    today = timer.today()
    start_time = timer.tic()
    for iter_i in range(start_iter, 1000000):
        # evaluation
        if iter_i > 0 and iter_i % args.eval_interval == 0:
            if not args.debug_mode:
                model.eval()
            with timer.contexttimer() as t0:
                model_eval = api.Detector(model_and_cfg=(model, global_cfg))
                dts = model_eval.evaluation_predict(
                    eval_info,
                    input_size=target_size,
                    conf_thres=ap_conf_thres,
                    catIdx2id=dataset.catIdx2id
                )
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
        # subdivision loop
        optimizer.zero_grad()
        for _ in range(subdivision):
            batch = dataset.get_next()
            imgs, labels = batch['images'], batch['labels']
            # np_im = image_ops.img_tensor_to_np(imgs[0], model.input_format, 'BGR_uint8')
            # labels[0].draw_on_np(np_im, class_map='ImageNet', imshow=True)
            imgs = imgs.cuda()
            # try:
            dts, loss = model(imgs, labels)
            assert not torch.isnan(loss)
            loss.backward()
            # except RuntimeError as e:
            #     print(e)
            #     if 'CUDA out of memory' in str(e):
            #         print(f'CUDA out of memory at imgsize={dataset.img_size},',
            #               f'batchsize={batch_size}')
            #         print('Trying to reduce the batchsize at that image size...')
            #         AUTO_BATCHSIZE[str(dataset.img_size)] -= 1
            #         dataset.to_iterator(batch_size=batch_size-1, shuffle=True,
            #                             num_workers=num_cpu, pin_memory=True)
            #     else:
            #         raise e
            # assert AUTO_BATCHSIZE[str(dataset.img_size)] == batch_size
            if global_cfg['train.hard_example_mining'] in {'probability'}:
                # calculate AP for each image
                idxs,img_ids,anns = batch['indices'],batch['image_ids'],batch['anns']
                for d, _idx, _id, g in zip(dts, idxs, img_ids, anns):
                    d: ImageObjects
                    d = d.post_process(conf_thres=ap_conf_thres,
                                       nms_thres=global_cfg['test.nms_thres'])
                    d = d.to_json(img_id=_id, eval_type=eval_info['eval_type'])
                    _, ap, ap50, ap75 = eval_func_(d, g, str_print=False)
                    dataset.update_ap(_idx, ap)

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0/subdivision)
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % args.print_interval == 0:
            sec_used  = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            _ai       = sec_used / (iter_i+1-start_iter)
            avg_iter  = timer.sec2str(_ai)
            avg_100img   = timer.sec2str(_ai / batch_size / subdivision * 100)
            avg_epoch = timer.sec2str(_ai / batch_size / subdivision * 118287)
            print(f'\nTotal time: {time_used}, 100 imgs: {avg_100img}, ',
                  f'iter: {avg_iter}, COCO epoch: {avg_epoch}')
            print(f'effective batch size = {batch_size} * {subdivision}')
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
            subdivision = int(np.ceil(super_batchsize / batch_size))
            dataset.img_size = imgsize
            dataset.to_iterator(batch_size=batch_size, shuffle=True,
                                num_workers=num_cpu, pin_memory=True)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                args.optimizer: optimizer.state_dict(),
                'dataset': dataset.hem_state
            }
            save_path = f'{PROJECT_ROOT}/weights/{job_name}_{today}_{iter_i}.pth'
            torch.save(state_dict, save_path)

        # save detection
        if iter_i > 0 and iter_i % args.demo_interval == 0:
            if not args.debug_mode:
                model.eval()
            model_eval = api.Detector(model_and_cfg=(model, global_cfg))
            demo_images_dir = f'{PROJECT_ROOT}/images/{args.demo_images}'
            for imname in os.listdir(demo_images_dir):
                # if not imname.endswith('.jpg'): continue
                impath = os.path.join(demo_images_dir, imname)
                np_img = model_eval.detect_one(img_path=impath, return_img=True,
                                        conf_thres=0.3, input_size=target_size)
                if args.debug_mode:
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


if __name__ == "__main__":
    main()
