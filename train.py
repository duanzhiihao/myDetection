# This is the main training file we are using
import os
import argparse
import functools
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

import sys
import json
from io import StringIO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.registry import name_to_model
from datasets import Dataset4ObjDet
from utils import timer, visualization
import api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='d1_345_a3_conf_yolo')
    parser.add_argument('--train_set', type=str, default='debug3')
    parser.add_argument('--val_set', type=str, default='debug3')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--checkpoint', type=str)

    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--res_min', type=int, default=384)
    parser.add_argument('--res_max', type=int, default=512)

    parser.add_argument('--print_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    parser.add_argument('--demo_interval', type=int, default=20)
    parser.add_argument('--demo_images_dir', type=str, default='./images/debug3/')
    
    parser.add_argument('--debug_mode', action='store_true')
    # parser.add_argument('--debug_mode', type=bool, default=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    print('Initialing model...')
    model, global_cfg = name_to_model(args.model)

    # -------------------------- settings ---------------------------
    target_size = round(args.resolution / 128) * 128
    job_name = f'{args.model}_{args.train_set}{target_size}'
    # multi-scale training setting
    enable_aug = not args.debug_mode
    multiscale = not args.debug_mode
    multiscale_interval = 10
    # dataloader setting
    batch_size = args.batch_size
    num_cpu = 0 if batch_size == 1 else 4
    subdivision = 128 // batch_size if not args.debug_mode else 3
    print(f'effective batch size = {batch_size} * {subdivision}')
    # optimizer setting
    decay_SGD = global_cfg['train.sgd.weight_decay'] * batch_size * subdivision
    base_lr = 0.0001 if args.debug_mode else 0.001
    lr_SGD = base_lr / batch_size / subdivision
    # Training set setting
    if args.train_set == 'debug3':
        training_set_cfg = {
            'img_dir': '../COCO/val2017',
            'json_path': '../COCO/annotations/debug3.json',
            'ann_bbox_format': 'x1y1wh',
            'input_image_format': model.input_format,
            'img_size': args.res_max,
            'enable_aug': enable_aug,
        }
    elif args.train_set == 'COCO':
        training_set_cfg = {
            'img_dir': '../COCO/train2017',
            'json_path': '../COCO/annotations/instances_train2017.json',
            'ann_bbox_format': 'x1y1wh',
            'input_image_format': model.input_format,
            'img_size': args.res_max,
            'enable_aug': enable_aug,
        }
    elif args.train_set == 'COCO80-R':
        # TODO
        dataset_cfg = {
            'train_img_dir': '../COCO/train2017',
            'train_json': '../COCO/annotations/instances_train2017.json',
            'val_img_dir': '../COCO/val2017',
            'val_json': '../COCO/annotations/instances_val2017.json',
            'rotated_bbox': True,
        }
    else:
        raise NotImplementedError()
    # Validation set setting
    if args.train_set == 'debug3':
        val_img_dir = './images/debug3/'
        val_json_path = '../COCO/annotations/debug3.json'
        validation_func = coco_evaluate_json
    elif args.val_set == 'val2017':
        val_img_dir = '../COCO/val2017'
        val_json_path = '../COCO/annotations/instances_val2017.json'
        validation_func = coco_evaluate_json
    else:
        raise NotImplementedError()
    
    # Prepare model
    pnum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters in {args.model}:', pnum)
    model = model.cuda()
    model.train()

    print(f'Initialing training set...')
    print(training_set_cfg)
    dataset = Dataset4ObjDet(training_set_cfg)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_cpu,
                            collate_fn=Dataset4ObjDet.collate_func, pin_memory=True)
    dataiterator = iter(dataloader)

    start_iter = -1
    if args.checkpoint:
        print("Loading checkpoint...", args.checkpoint)
        weights_path = os.path.join('./weights/', args.checkpoint)
        previous_state = torch.load(weights_path)
        model.load_state_dict(previous_state['model'])
        start_iter = previous_state['iter'] + 1
        print(f'Start from iteration: {start_iter}')

    print('Initialing tensorboard SummaryWriter...')
    if args.debug_mode:
        logger = SummaryWriter(f'./logs/debug/{job_name}')
    else:
        logger = SummaryWriter(f'./logs/{job_name}')

    print(f'Initialing optimizer with lr: {lr_SGD}, decay: {decay_SGD}')
    params = []
    # set weight decay only on conv.weight
    for key, value in model.named_parameters():
        if 'conv' in key:
            params += [{'params':value, 'weight_decay':decay_SGD}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]
    # Initialize optimizer
    optimizer = torch.optim.SGD(params, lr=lr_SGD, momentum=0.9, dampening=0,
                                weight_decay=decay_SGD)
    if args.checkpoint and 'optimizer' in previous_state:
        optimizer.load_state_dict(previous_state['optimizer'])
    # Learning rate scheduler
    warmup_iter = 100 if args.debug_mode else 1000
    lr_schedule_func = functools.partial(lr_warmup, warm_up=warmup_iter)
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_schedule_func, last_epoch=start_iter)

    print('Starting training...')
    today = timer.today()
    start_time = timer.tic()
    for iter_i in range(start_iter, 100000):
        # evaluation
        if iter_i > 0 and iter_i % args.eval_interval == 0:
            with timer.contexttimer() as t0:
                model_eval = api.Detector(model_and_cfg=(model, global_cfg))
                dts = model_eval.predict_imgDir(val_img_dir, input_size=target_size,
                                                to_square=True, conf_thres=0.005)
                eval_str, ap, ap50, ap75 = validation_func(dts, val_json_path)
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

        # subdivision loop
        optimizer.zero_grad()
        for _ in range(subdivision):
            try:
                imgs, labels, imid, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, labels, imid, _ = next(dataiterator)  # load a batch
            # visualization.imshow_tensor(imgs)
            imgs = imgs.cuda()
            loss = model(imgs, labels)
            loss.backward()
            # if args.adversarial:
            #     imgs = imgs + imgs.grad*0.05
            #     imgs = imgs.detach()
            #     # visualization.imshow_tensor(imgs)
            #     loss = model(imgs, targets)
            #     loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % args.print_interval == 0:
            sec_used = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            avg_iter = timer.sec2str(sec_used/(iter_i+1-start_iter))
            avg_img = avg_iter / batch_size / subdivision
            avg_epoch = avg_img * 118287
            print(f'\nTotal time: {time_used}, 100 imgs: {avg_img*100}, ',
                  f'iter: {avg_iter}, epoch: {avg_epoch}')
            current_lr = scheduler.get_last_lr()[0] * batch_size * subdivision
            print(f'[Iteration {iter_i}] [learning rate {current_lr:.3g}]',
                  f'[Total loss {loss:.2f}] [img size {dataset.img_size}]')
            print(model.loss_str)
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print(f'Max GPU memory usage: {max_cuda} GigaBytes')
        torch.cuda.reset_max_memory_allocated(0)

        # random resizing
        if multiscale and iter_i > 0 and (iter_i % multiscale_interval == 0):
            Rmin, Rmax = round(args.res_min / 128), round(args.res_max / 128)
            imgsize = random.randint(Rmin, Rmax) * 128
            dataset.img_size = imgsize
            dataloader = DataLoader(dataset, batch_size, True, num_workers=num_cpu,
                            collate_fn=Dataset4ObjDet.collate_func, pin_memory=True)
            dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', f'{job_name}_{today}_{iter_i}.pth')
            torch.save(state_dict, save_path)

        # save detection
        if iter_i % args.demo_interval == 0:
            model_eval = api.Detector(model_and_cfg=(model, global_cfg))
            for imname in os.listdir(args.demo_images_dir):
                if not imname.endswith('.jpg'): continue
                impath = os.path.join(args.demo_images_dir, imname)
                np_img = model_eval.detect_one(img_path=impath, return_img=True,
                                               conf_thres=0.3)
                if args.debug_mode:
                    cv2_im = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    log_dir = f'./logs/{args.model}_debug/'
                    if not os.path.exists(log_dir): os.mkdir(log_dir)
                    s = os.path.join(log_dir, f'{imname[:-4]}_iter{iter_i}.jpg')
                    cv2.imwrite(s, cv2_im)
                else:
                    logger.add_image(impath, np_img, iter_i, dataformats='HWC')
            model.train()


# Learning rate setup
def lr_warmup(i, warm_up=1000):
    if i < warm_up:
        factor = i / warm_up
    elif i < 40000:
        factor = 1
    elif i < 80000:
        factor = 0.4
    elif i < 100000:
        factor = 0.1
    elif i < 200000:
        factor = 1
    else:
        factor = 0.01
    return factor


def coco_evaluate_json(dts_json, gt_json_path):
    json.dump(dts_json, open('./tmp.json','w'), indent=1)
    print('Initialing validation set...')
    cocoGt = COCO(gt_json_path)
    cocoDt = cocoGt.loadRes('./tmp.json')
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    # have to manually get the evaluation string 
    stdout = sys.stdout
    s = StringIO()
    sys.stdout = s
    cocoEval.summarize()
    sys.stdout = stdout
    s.seek(0)
    s = s.read()
    print(s)
    ap, ap50, ap75 = cocoEval.stats[0], cocoEval.stats[1], cocoEval.stats[2]
    return s, ap, ap50, ap75


if __name__ == "__main__":
    main()
