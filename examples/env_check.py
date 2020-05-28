
def check_settings():
    import os
    from settings import COCO_DIR, COSSY_DIR
    if not os.path.exists(COCO_DIR):
        print('Cannot find the COCO dataset')
    if not os.path.exists(COSSY_DIR):
        print('Cannot find the COSSY dataset')


if __name__ == "__main__":
    check_settings()