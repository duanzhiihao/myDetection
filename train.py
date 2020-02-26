# This is the main training file we are using
import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

from datasets import Dataset4ObjDet
from utils import timer, visualization
import api


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov3')
    parser.add_argument('--backbone', type=str, default='dark53')
    parser.add_argument('--dataset', type=str, default='COCO')
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--img_norm', action='store_true')

    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--res_min', type=int, default=384)
    parser.add_argument('--res_max', type=int, default=640)

    parser.add_argument('--checkpoint', type=str)

    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--img_interval', type=int, default=500)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    
    parser.add_argument('--debug', action='store_true') # default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert torch.cuda.is_available()
    # -------------------------- settings ---------------------------
    target_size = round(args.resolution / 128) * 128
    job_name = f'{args.model}_{args.backbone}_{args.dataset}{target_size}'
    # multi-scale training setting
    enable_aug = True
    multiscale = True
    multiscale_interval = 10
    # dataloader setting
    batch_size = args.batch_size
    num_cpu = 0 if batch_size == 1 else 4
    subdivision = 128 // batch_size
    print(f'effective batch size = {batch_size} * {subdivision}')
    # optimizer setting
    decay_SGD = 0.0005 * batch_size * subdivision
    lr_SGD = 0.001 / batch_size / subdivision
    # Learning rate setup
    def lr_schedule_func(i):
        warm_up = 1000
        if i < warm_up:
            factor = i / warm_up
        elif i < 40000:
            factor = 1.0
        elif i < 80000:
            factor = 0.5
        elif i < 100000:
            factor = 0.2
        elif i < 200000:
            factor = 0.1
        else:
            factor = 0.01
        return factor
    # dataset setting
    # only_person = False if args.model == 'Hitachi80' else True
    # print('Only train on person images and object:', only_person)
    if args.dataset == 'COCO':
        import sys
        import json
        from io import StringIO
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        train_img_dir = '../COCO/train2017/'
        train_json = '../COCO/annotations/instances_train2017.json'
        val_img_dir = '../COCO/val2017/'
        valjson = '../COCO/annotations/instances_val2017.json'
        def evaluate_json(dts_json):
            json.dump(dts_json, open('./tmp.json','w'), indent=1)
            print('Initialing validation set...')
            cocoGt = COCO(valjson)
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
    
    print('Initialing training set...')
    dataset = Dataset4ObjDet(train_img_dir, train_json, 'x1y1wh', img_size=args.res_max, 
                             augmentation=True, debug_mode=args.debug)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
    dataiterator = iter(dataloader)
    
    eval_img_names = os.listdir('./images/')
    eval_img_paths = [os.path.join('./images/',s) for s in eval_img_names]

    print('Initialing model...')
    if args.model == 'yolov3':
        from models.yolov3 import YOLOv3
        model = YOLOv3(backbone=args.backbone, img_norm=False)
    else:
        raise Exception('Unknown madel name')
    
    model = model.cuda()
    model.train()

    start_iter = -1
    if args.checkpoint:
        print("Loading ckpt...", args.checkpoint)
        weights_path = os.path.join('./weights/', args.checkpoint)
        previous_state = torch.load(weights_path)
        model.load_state_dict(previous_state['model'])
        start_iter = previous_state['iter'] + 1
        print(f'Start from iteration: {start_iter}')

    logger = SummaryWriter(f'./logs/{job_name}')

    # optimizer setup
    params = []
    # set weight decay only on conv.weight
    for key, value in model.named_parameters():
        if 'conv' in key:
            params += [{'params':value, 'weight_decay':decay_SGD}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]

    optimizer = torch.optim.SGD(params, lr=lr_SGD, momentum=0.9, dampening=0,
                                weight_decay=decay_SGD)
    if args.checkpoint and 'optimizer' in previous_state:
        optimizer.load_state_dict(previous_state['optimizer'])
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_schedule_func, last_epoch=start_iter)

    print('Starting training...')
    today = timer.today()
    start_time = timer.tic()
    for iter_i in range(start_iter, 500000):
        # evaluation
        if iter_i > 0 and iter_i % args.eval_interval == 0:
            with timer.contexttimer() as t0:
                model_eval = api.Detector(model=model)
                dts = model_eval.predict_imgDir(val_img_dir, input_size=target_size,
                                                conf_thres=0.005)
                eval_str, ap, ap50, ap75 = evaluate_json(dts)
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
                imgs, labels, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, labels, _, _ = next(dataiterator)  # load a batch
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
            avg_epoch = avg_iter / batch_size / subdivision * 118287
            print(f'\nTotal time: {time_used}, iter: {avg_iter}, epoch: {avg_epoch}')
            current_lr = scheduler.get_last_lr()[0] * batch_size * subdivision
            print(f'[Iteration {iter_i}] [learning rate {current_lr:.3g}]',
                  f'[Total loss {loss:.2f}] [img size {dataset.img_size}]')
            print(model.loss_str)
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print(f'Max GPU memory usage: {max_cuda} GigaBytes')
            torch.cuda.reset_max_memory_allocated(0)
            # if hasattr(model, 'time_monitor'):
            #     print(model.time_monitor())

        # random resizing
        if multiscale and iter_i > 0 and (iter_i % multiscale_interval == 0):
            Rmin, Rmax = round(args.res_min / 128), round(args.res_max / 128)
            imgsize = random.randint(Rmin, Rmax) * 128
            dataset.img_size = imgsize
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_cpu, pin_memory=True, drop_last=False)
            dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', f'{job_name}_{today}_{iter_i}.ckpt')
            torch.save(state_dict, save_path)

        # save detection
        if iter_i > 0 and iter_i % args.img_interval == 0:
            for impath in eval_img_paths:
                model_eval = api.Detector(model=model)
                eval_img = Image.open(impath)
                dts = model_eval.detect_one(pil_img=eval_img, input_size=target_size,
                                            conf_thres=0.1, visualize=False)

                np_img = np.array(eval_img)
                visualization.draw_cocobb_on_np(np_img, dts)
                np_img = cv2.resize(np_img, (416,416))
                logger.add_image(impath, np_img, iter_i, dataformats='HWC')
            model.train()