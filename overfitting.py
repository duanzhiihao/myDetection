# Overfit on one image for debugging
import os
import random
import torch
from torch.utils.data import DataLoader
import cv2

from models.registry import name_to_model
from datasets import Dataset4ObjDet
from utils import timer, visualization
import api


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    # -------------------------- settings ---------------------------
    model_name = 'fcs_d53yc3_sl1'
    # model_name = 'yv3_ltrb'
    target_size = 512
    batch_size = 2
    subdivision = 1
    print(f'effective batch size = {batch_size} * {subdivision}')
    # optimizer setting
    decay_SGD = 0.0005 * batch_size * subdivision
    lr_SGD = 0.0001 / batch_size / subdivision
    # Dataset setting
    train_img_dir = '../COCO/val2017/'
    train_json = '../COCO/annotations/debug3.json'
    
    print('Initialing training set...')
    dataset = Dataset4ObjDet(train_img_dir, train_json, 'x1y1wh', img_size=target_size, 
                             augmentation=False, debug_mode=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=Dataset4ObjDet.collate_func, num_workers=0, pin_memory=True)
    dataiterator = iter(dataloader)

    print('Initialing model...')
    model = name_to_model(model_name)
    model = model.cuda()
    model.train()

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
    # optimizer = torch.optim.SGD(params, lr=lr_SGD)

    print('Starting training...')
    today = timer.today()
    start_time = timer.tic()
    for iter_i in range(5000):
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
        optimizer.step()

        # logging
        if iter_i % 10 == 0:
            print(f'[Iteration {iter_i}]',
                  f'[Total loss {loss:.2f}] [img size {dataset.img_size}]')
            print(model.loss_str)

        # random resizing
        # if False and iter_i > 0 and (iter_i % 10 == 0):
        #     Rmin, Rmax = round(args.res_min / 128), round(args.res_max / 128)
        #     imgsize = random.randint(Rmin, Rmax) * 128
        #     dataset.img_size = imgsize
        #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
        #                         num_workers=0, pin_memory=True, drop_last=False)
        #     dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % 1000 == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', f'debug.ckpt')
            torch.save(state_dict, save_path)

        # show detection
        if iter_i > 0 and iter_i % 20 == 0:
            for imid in dataset.img_ids:
                imname = dataset.imgid2info[imid]['file_name']
                impath = os.path.join(train_img_dir, imname)
                model_eval = api.Detector(model=model)
                np_img = model_eval.detect_one(img_path=impath, return_img=True,
                                            input_size=target_size, conf_thres=0.3)
                cv2_im = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                log_dir = f'./logs/{model_name}_debug/'
                if not os.path.exists(log_dir): os.mkdir(log_dir)
                s = os.path.join(log_dir, f'img{imid}_iter{iter_i}.jpg')
                cv2.imwrite(s, cv2_im)
            model.train()