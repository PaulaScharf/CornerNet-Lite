import os
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm
import cv2
import json

def convert_labels(org, dest, split, names):
    print('[INFO] converting labels...')
    output = {
                "info": {
                    "description": "Converted "+split,
                    "url": "http:\\/\\/mscoco.org",
                    "version": "1.0",
                    "year": 2022,
                    "contributor": "Paula Scharf",
                    "date_created": "2022-10-20 09:11:52.357475"
                },
                "type": "instances",
                "images": [],
                "annotations": [],
                "categories": []
            }
    for i, name in enumerate(names):
        output['categories'].append({
            "supercategory": "insect",
            "id": i+1,
            "name": name
        })
    
    org_img_dir = org
    org_lab_dir = org.replace("images", "labels")
    src_files = os.listdir(org_img_dir)
    pbar = enumerate(src_files)
    pbar = tqdm(pbar, total=len(src_files))  # progress bar
    for i, file_name in pbar:
        full_img_path = os.path.join(org_img_dir, file_name)
        img = cv2.imread(full_img_path)
        if img is None:
            img_shape = img.shape
        else:
            img_shape = [3456,4608]
        output['images'].append({
            "license": 4,
            "url": "http:\\/\\/farm7.staticflickr.com\\/6116\\/6255196340_da26cf2c9e_z.jpg",
            "file_name": file_name,
            "height": img_shape[1],
            "width": img_shape[0],
            "date_captured": "2022-10-20 09:11:52.357475",
            "id": i
        })

        full_label_path = os.path.join(org_lab_dir, Path(file_name).with_suffix('.txt'))
        if os.stat(full_label_path).st_size > 0:
            with open(full_label_path) as f:
                lines = f.readlines()
                for line in lines:
                    line_arr = line.split()
                    y_org = int(float(line_arr[1])*img_shape[1])
                    x_org = int(float(line_arr[2])*img_shape[0])
                    height = int((float(line_arr[3])*img_shape[1])/2)
                    width = int((float(line_arr[4])*img_shape[0])/2)
                    output['annotations'].append({
                        "area": (width*2)*(height*2),
                        "iscrowd": 0,
                        "image_id": i,
                        "bbox": [
                            y_org-height,
                            x_org-width,
                            height*2,
                            width*2
                        ],
                        "category_id": int(line_arr[0])+1,
                        "id": 0
                    })
    
    split_dest = os.path.join(dest, split+".json")

    with open(split_dest, 'w') as f:
        json.dump(output, f)
    
    return split_dest

def move_images(org, dest, split):
    print('[INFO] copying images from source...')
    split_dest = os.path.join(dest, split)
    os.mkdir(split_dest)


    src_files = os.listdir(org)
    pbar = enumerate(src_files)
    pbar = tqdm(pbar, total=len(src_files))  # progress bar
    for _, file_name in pbar:
        full_file_name = os.path.join(org, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, split_dest)

    return split_dest

def convert_yolo_coco(data_yaml, dest_dir):
    dest_img_dir = os.path.join(dest_dir, "images")
    dest_anno_dir = os.path.join(dest_dir, "annotations")
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
        os.mkdir(dest_img_dir)
        os.mkdir(dest_anno_dir)
    elif not os.path.isdir(dest_img_dir) and not os.path.isdir(dest_anno_dir):
        os.mkdir(dest_img_dir)
        os.mkdir(dest_anno_dir)
    else:
        raise Exception("Destination folder should be empty")

    print('[INFO] reading data.yaml...')
    with open(data_yaml) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    train_path = data_dict['train']
    val_path = data_dict['val']
    test_path = data_dict['test']
    names = data_dict['names']

    split = "train"
    print('[INFO] '+split+'...')
    res_train_imgs = move_images(train_path, dest_img_dir, split)
    res_train_annos = convert_labels(train_path, dest_anno_dir, split, names)
    split = "val"
    print('[INFO] '+split+'...')
    res_val_imgs = move_images(val_path, dest_img_dir, split)
    res_val_annos = convert_labels(val_path, dest_anno_dir, split, names)
    split = "test"
    print('[INFO] '+split+'...')
    res_test_imgs = move_images(test_path, dest_img_dir, split)
    res_test_annos = convert_labels(test_path, dest_anno_dir, split, names)

    return res_train_imgs, res_val_imgs, res_test_imgs, res_train_annos, res_val_annos, res_test_annos

convert_yolo_coco("/scratch/tmp/p_scha35/yolo-test/data/yolo_normal/data.yaml", "/scratch/tmp/p_scha35/yolo-test/CornerNet-Lite/data/custom")