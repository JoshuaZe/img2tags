import numpy as np
from PIL import Image
from colorthief import MMCQ
from img2tags.saliency_model import generate_attention_mask
from img2tags.nearest_color import find_nearest_color


def extract_main_color(fp, df_color_dict, color_count=8, quality=10, th_color_ratio=0.2, dist_type='CIE94'):
    image_pil = Image.open(fp).convert('RGB')
    # get attention mask
    image_arr = np.array(image_pil)
    image_arr = image_arr[:, :, ::-1].copy()
    mask = generate_attention_mask(image_arr)
    # get color plate
    # https://github.com/fengsp/color-thief-py
    # https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
    # https://www.hisour.com/zh/color-appearance-model-24824/
    image_pil_rgba = Image.fromarray(
        np.dstack((np.array(image_pil) * mask[:, :, np.newaxis], mask * 255)).astype('uint8'), 'RGBA'
    )
    width, height = image_pil_rgba.size
    pixels = image_pil_rgba.getdata()
    pixel_count = width * height
    valid_pixels = []
    for i in range(0, pixel_count, quality):
        r, g, b, a = pixels[i]
        # If pixel is mostly opaque
        if a >= 125:
            valid_pixels.append((r, g, b))
    # Send array to quantize function which clusters values
    # using median cut algorithm
    cmap = MMCQ.quantize(valid_pixels, color_count)
    # ranking
    cmap.vboxes.sort_key = lambda x: -x['vbox'].count
    cmap.vboxes.sort()
    pixel_count_by_color = cmap.vboxes.map(lambda x: x['vbox'].count / len(valid_pixels))
    num_dominent_colors = max(sum(np.where(np.array(pixel_count_by_color) >= th_color_ratio, 1, 0)), 1)
    dominent_palette = cmap.palette[:num_dominent_colors]
    # generate name list
    lst_main_color = []
    for each_color in dominent_palette:
        lst_main_color.extend(find_nearest_color(each_color, df_color_dict, dist_type=dist_type)['color_names'])
    # return [find_nearest_color(each_color, df_color_dict, dist_type=dist_type) for each_color in dominent_palette]
    return list(set(lst_main_color))


if __name__ == '__main__':
    import os
    import pandas as pd

    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    PROJ_DIR = os.path.abspath(os.path.join(APP_DIR, '..'))
    COLOR_DICT_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/color_dictionary.csv'))

    df_color_dict = pd.read_csv(COLOR_DICT_PATH)

    img_colors = extract_main_color(
        '/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000002_112837.jpg',
        df_color_dict
    )
    # image = Image.open(
    #     '/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000002_112837.jpg'
    # ).convert('RGB').show()
    print(img_colors)
