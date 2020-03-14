

def name_to_model(model_name):
    if model_name == 'yolov3':
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False)
    elif model_name == 'yv3_c3_af':
        from .yolov3 import YOLOv3
        return YOLOv3(class_num=80, backbone='dark53', img_norm=False,
                      pred_layer='FCOS')
    else:
        raise Exception('Unknown model name')
