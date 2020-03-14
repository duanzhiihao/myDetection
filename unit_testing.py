# This file is for debugging
import numpy as np
import matplotlib.pyplot as plt
import torch

def test_training():
    from tqdm import tqdm
    import utils.visualization as visUtils
    from datasets import Dataset4ObjDet
    import torchvision.transforms.functional as tvf
    img_dir = '../COCO/train2017'
    ann_json = '../COCO/annotations/instances_train2017.json'
    # img_dir = '../COCO/val2017'
    # ann_json = '../COCO/annotations/instances_val2017.json'
    
    dataset = Dataset4ObjDet(img_dir=img_dir, json_path=ann_json, bb_format='x1y1wh',
                             img_size=512, augmentation=True,
                             debug_mode=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, 
                            num_workers=0, pin_memory=True, drop_last=False)
    dataiterator = iter(dataloader)

    for _ in range(10):
        try:
            img, labels, img_id, _ = next(dataiterator)
        except StopIteration:
            dataiterator = iter(dataloader)
            img, labels, img_id, _ = next(dataiterator)  # load a batch
        # print(img_id)
        img, labels = img.squeeze(), labels.squeeze(0)
        img = tvf.to_pil_image(img)
        img = np.array(img)
        labels[:,1:5] *= img.shape[0]
        gt_num = (labels[:,:4].squeeze().sum(dim=1) > 0).sum().item()
        visUtils.draw_cocobb_on_np(img, labels[:gt_num,:], bb_type='gtbb',
                                   print_dt=False)
        plt.figure(figsize=(8,8))
        plt.imshow(img)
        plt.show()
        debug = 1

    debug = 1
    
    # from models.yolov3 import YOLOv3
    # model = YOLOv3(80, 'dark53', img_norm=False)

    # # torch.cuda.reset_max_memory_allocated()
    # img, labels, _, _ = next(dataiterator)
    # oupu = model(img, labels)
    # oupu.backward()

    debug = 1


def test_iou():
    import numpy as np
    from time import time
    from utils.iou_mask import iou_mask, iou_geometry, iou_rle
    torch.random.manual_seed(10)
    
    boxes1 = torch.rand(50,5)
    boxes1[:,0:2] = -0.1 + boxes1[:,0:2]
    boxes1[:,2:4] = -0.1 + boxes1[:,2:4]
    # boxes1[:,0:4] *= 1024
    boxes1[:,4] *= 180
    boxes2 = torch.rand(100,5)
    boxes2[:,0:2] = 0.3 + boxes2[:,0:2]*0.4
    boxes2[:,2:4] = 0.3 + boxes2[:,2:4]*0.4
    # boxes2[:,0:4] *= 1024
    boxes2[:,4] *= 180

    start = time()
    ious1 = iou_rle(boxes1, boxes2, xywha=True, is_degree=True, img_size=1024,
                    normalized=True)
    print(f'rle ellapsed time: {time()-start:.3f}s')
    # print('mean of IoU using rle:', ious1.mean())

    start = time()
    ious2 = iou_mask(boxes1, boxes2, xywha=True, mask_size=128, is_degree=True)
    print(f'mask ellapsed time: {time()-start:.3f}s')
    # print('mean of IoU using mask:', ious2.mean())

    start = time()
    ious3 = iou_geometry(boxes1, boxes2, xywha=True, is_degree=True)
    print(f'geo ellapsed time: {time()-start:.3f}s')
    # print('mean of IoU using geometry:', ious3.mean())

    print('MAE rle:', np.abs(ious1-ious3).mean())
    print('MAE mask:', np.abs(ious2-ious3).mean())

    debug = 1

    # ious = ious.cpu().flatten().numpy()
    # plt.hist(ious, bins=100)
    # plt.show()


def print_model():
    from models import YOLOv3
    net = YOLOv3()
    with open('./print_weights.txt', 'w') as f:
        state_dict = net.state_dict()
        for k,v in state_dict.items():
            print(k, file=f)
    with open('./print_model.txt', 'w') as f:
        print(net, file=f)

    # net = load_weights_from(net, weights_path)

    # torch.save(net.state_dict(), weights_path)


def test_official():
    from collections import OrderedDict

    from models.yolov3 import YOLOv3
    model = YOLOv3(class_num=80)
    ckpt = torch.load('./weights/yolov3_official.pth')
    model.load_state_dict(ckpt['model'])

    # weights = torch.load('./weights/yolov3_official.pth')

    # with open('my.txt', 'r') as f:
    #     names = f.readlines()
    
    # new_weights = {'model': OrderedDict()}
    # for i, (k,v) in enumerate(weights.items()):
    #     new_name = names[i].strip().split(' ')[0]
    #     new_weights['model'][new_name] = v
    
    # model.load_state_dict(new_weights['model'])
    # ckpt = {'model': model.state_dict()}
    # torch.save(ckpt, './weights/my_official.pth')


def cpuvsgpu():
    import torch
    import time
    from tqdm import tqdm

    # warm up
    nG = 128
    x_shift = torch.arange(nG, dtype=torch.float).repeat(nG,1).view(1,1,nG,nG)
    y_shift = torch.arange(nG, dtype=torch.float).repeat(nG,1).t().view(1,1,nG,nG)
    x_shift = torch.arange(nG, dtype=torch.float,
                        device='cuda').repeat(nG,1).view(1,1,nG,nG)
    y_shift = torch.arange(nG, dtype=torch.float,
                        device='cuda').repeat(nG,1).t().view(1,1,nG,nG)

    ### CPU
    # start_time = time.time()
    # a = torch.ones(100,100,100)
    # for _ in range(10000):
    #     a += a
    # elapsed_time = time.time() - start_time
    # print('CPU time = ',elapsed_time)

    ### GPU
    # start_time = time.time()
    # b = torch.ones(100,100,100).cuda()
    # for _ in range(10000):
    #     b += b
    # elapsed_time = time.time() - start_time
    # print('GPU time = ',elapsed_time)

    ### Transfer time
    # start_time = time.time()
    # t = torch.ones(1,1,1)
    # num = 10000
    # for _ in tqdm(range(num)):
    #     t = t.cuda()
    #     t = t.cpu()
    # elapsed_time = time.time() - start_time
    # print('Average time = ', elapsed_time/num)
    
    ### Transfer time
    nG = 128
    torch.cuda.synchronize()
    
    start_time = time.time()
    x_shift = torch.arange(nG, dtype=torch.float).repeat(nG,1).view(1,1,nG,nG)
    y_shift = torch.arange(nG, dtype=torch.float).repeat(nG,1).t().view(1,1,nG,nG)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print('CPU time = ', elapsed_time)

    start_time = time.time()
    x_shift = torch.arange(nG, dtype=torch.float,
                        device='cuda').repeat(nG,1).view(1,1,nG,nG)
    y_shift = torch.arange(nG, dtype=torch.float,
                        device='cuda').repeat(nG,1).t().view(1,1,nG,nG)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print('GPU time = ', elapsed_time)


def adversarial():
    from models.exact import YOLOv3
    # anchors = [
    #     [10.7, 21.888], [21.6448, 40.0672], [32.9536, 66.9408],
    #     [39.0944, 88.2208], [46.0256, 108.3456], [69.1296, 103.36],
    #     [59.4016, 124.5792], [93.2064, 125.0656], [78.128, 158.4448]
    # ]
    # indices = [[6,7,8], [3,4,5], [0,1,2]]
    model = YOLOv3(loss_angle='period_L2')
    weights_path = './weights/angle_exact_old_COCO_advTrue_Nov18_78000.ckpt'
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model = model.cuda()
    model.train()

    from datasets import Dataset4YoloAngle
    img_dir = '../../../COSSY/Lecture/'
    ann_json = '../../../COSSY/annotations/Lecture.json'
    dataset = Dataset4YoloAngle(img_dir=img_dir, json_path=ann_json,
                                img_size=800, augmentation=False,
                                debug_mode=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
                            num_workers=0, pin_memory=True, drop_last=False)
    from utils import utils, visualization
    import matplotlib.pyplot as plt
    import cv2
    
    for imgs, labels, _, info in dataloader:
        ori_imgs = imgs.clone().cuda()
        labels = labels.cuda()

        with torch.no_grad():
            dts = model(ori_imgs).detach().squeeze()
        dts = dts[dts[:,5] >= 0.1].cpu()
        dts = utils.nms(dts, is_degree=True, nms_thres=0.45)
        ori_np_img = visualization.tensor_to_npimg(ori_imgs)
        visualization.draw_dt_on_np(ori_np_img, dts, print_dt=False, text_size=0.5)
        plt.figure()
        plt.imshow(ori_np_img)
        cv2.imwrite('./results/initial.png', ori_np_img[:,:,::-1]*255)

        # gradients = imgs.grad.clone().cpu()
        # b = torch.linspace(-1, 1, steps=100).numpy()
        # h = torch.histc(gradients, bins=100, min=-1, max=1).numpy() / 100
        # plt.figure()
        # plt.bar(b, h)
    
        # mask = torch.zeros(1,3,800,800).cuda()
        # mask[:,:,300:500,300:500] = 1
        imgs = imgs.cuda()
        for i in range(20):
            imgs = imgs.detach().requires_grad_(True)
            loss = model(imgs, labels)
            loss.backward()
            if i == 0 or i == 9:
                # draw 'attention' heat map
                sensitive = imgs.grad.clone().detach().squeeze().permute(1,2,0).cpu().numpy()
                # plt.figure()
                # plt.imshow(sensitive)
                # cv2.imwrite(f'./results/iter_{i}.png', sensitive[:,:,::-1]*255)
            imgs = imgs + imgs.grad*0.05
        
        print('MAE:', (imgs - ori_imgs).abs().mean().cpu().item())
        with torch.no_grad():
            dts = model(imgs).detach().squeeze()
        dts = dts[dts[:,5] >= 0.1].cpu()
        dts = utils.nms(dts, is_degree=True, nms_thres=0.45)
        np_img = visualization.tensor_to_npimg(imgs.detach())
        visualization.draw_dt_on_np(np_img, dts, print_dt=False, text_size=0.5)
        plt.figure()
        plt.imshow(np_img)
        cv2.imwrite('./results/final.png', np_img[:,:,::-1]*255)

        plt.show()


if __name__ == "__main__":
    test_training()
    # test_iou()
    # print_model()
    # test_official()
    # cpuvsgpu()
    # adversarial()
    # test_efficient()
    # test_hist_equal()
    # test_tvf()