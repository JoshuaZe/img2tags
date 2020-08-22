import os
import json
import re
import string
from PIL import Image
import numpy as np
import pandas as pd


dataset = {
    "info": {},
    "images": [],
    "annotations": [],
    "categories": [],
    "licenses": []
}

dataset['categories'].append({
    'id': 1,
    'name': "short_sleeved_shirt",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 2,
    'name': "long_sleeved_shirt",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 3,
    'name': "short_sleeved_outwear",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 4,
    'name': "long_sleeved_outwear",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 5,
    'name': "vest",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 6,
    'name': "sling",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 7,
    'name': "shorts",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 8,
    'name': "trousers",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 9,
    'name': "skirt",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 10,
    'name': "short_sleeved_dress",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 11,
    'name': "long_sleeved_dress",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 12,
    'name': "vest_dress",
    'supercategory': "clothes"
})
dataset['categories'].append({
    'id': 13,
    'name': "sling_dress",
    'supercategory': "clothes"
})

if __name__ == '__main__':
    # input
    image_folder = "data/B/train/image"
    # img_id,img_name,img_url,source
    image_meta_path = "data/B/train/img_info.csv"
    df_image_meta = pd.read_csv(image_meta_path, keep_default_na=False)
    # img_id,obj_id,item_id,bbox,category_id,category_name,color_tags,attribute_tags
    annotation_path = "data/B/train/obj_info.csv"
    df_annotation = pd.read_csv(annotation_path, keep_default_na=False)
    # output
    coco_json_path = "data/B/train/train_coco.json"

    # processing to COCO format
    instance_id = 0
    for img_idx, image_row in df_image_meta.iterrows():
        img_id = image_row['img_id']
        img_name = image_row['img_name']
        img_url = image_row['img_url']
        source = image_row['source']
        # image meta info
        img_address = os.path.join(image_folder, img_name)
        image = Image.open(img_address)
        width, height = image.size
        dataset['images'].append({
            'id': img_idx,
            'width': width,
            'height': height,
            'file_name': img_name,
            'license': 0,
            'flickr_url': '',
            'coco_url': '',
            'date_captured': '',
            'source': source
        })
        # annotations of image
        df_image_annotation = df_annotation.query('img_id == "{}"'.format(img_id))
        for _, annotation_row in df_image_annotation.iterrows():
            obj_id = annotation_row['obj_id']
            item_id = annotation_row['item_id']
            bbox = list(map(float, re.split('[,]', annotation_row['bbox'])))
            category_id = annotation_row['category_id']
            category_name = annotation_row['category_name']
            color_tags = []
            if annotation_row['color_tags'] != "":
                color_tags = list(re.split('[,]', annotation_row['color_tags']))
            attribute_tags = []
            if annotation_row['attribute_tags'] != "":
                attribute_tags = list(re.split('[,]', annotation_row['attribute_tags']))
            # bbox info
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x_1 = bbox[0]
            y_1 = bbox[1]
            bbox_coco = [x_1, y_1, w, h]
            dataset['annotations'].append({
                'id': instance_id,
                'image_id': img_idx,
                'category_id': category_id,
                'item_id': item_id,
                'area': w * h,
                'bbox': bbox_coco,
                'color_tags': color_tags,
                'attribute_tags': attribute_tags
            })
            instance_id = instance_id + 1

    with open(coco_json_path, 'w') as f:
        json.dump(dataset, f)
