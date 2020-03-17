

def name_to_model(model_name):
    if model_name == 'yolov3':
        # darknet-53, YOLO fpn C3, 3x3 anchor boxes, 
        # xywh, norm by anchor, exp, xy:BCE, wh:L2
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False)

    elif model_name == 'yv3_ltrb':
        # YOLOv3, only change the xywh to ltrb
        # darknet-53, fpn C3, 3x3 anchor boxes, ltrb, norm by anchor, exp, L2
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False,
                      pred_layer='FCOS', ltrb='exp_l2')

    elif model_name == 'yv3_ltrb_sl1':
        # darknet-53, YOLO fpn C3, 3x3 anchor boxes, 
        # ltrb, norm by anchor, exp, smooth_L1
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False,
                      pred_layer='FCOS', ltrb='exp_sl1')

    elif model_name == 'yv3_ltrb_RAG':
        # darknet-53, YOLO fpn C3, 3x3 anchor boxes, ltrb, norm by anchor, GIoU
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False,
                      pred_layer='FCOS', ltrb='relu_ach_giou')
    
    elif model_name == 'fcs_d53yc3_sl1':
        # darknet-53, YOLO fpn C3, 3 strides,
        # ltrb, norm by stride, exp, smooth_L1, center 1.5 stride, no centerness
        from .fcos import FCOS
        return FCOS(backbone='dark53', fpn='yolo3_1anch', ltrb_setting='exp_sl1',
                    img_norm=False)

    else:
        raise Exception('Unknown model name')
