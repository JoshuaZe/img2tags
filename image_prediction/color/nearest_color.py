import pandas as pd
import math
import re
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994


df_color_dict = pd.read_csv('./color_dictionary.csv')


def rgb_to_hex(rgb: tuple):
    return '#%02X%02X%02X' % rgb


def hex_to_rgb(hex_code: str):
    hex_value = hex_code.lstrip('#')
    hex_len = len(hex_value)
    return tuple(int(hex_value[i:i + hex_len // 3], 16) for i in range(0, hex_len, hex_len // 3))


def color_distance(rgb_a: sRGBColor, rgb_b: sRGBColor, dist_type: str = 'sRGB'):
    # https://en.wikipedia.org/wiki/Color_difference
    func_switcher = {
        'CIE76': delta_e_cie1976,
        'CIE94': lambda color1, color2: delta_e_cie1994(color1, color2, K_L=2, K_C=1, K_H=1, K_1=0.048, K_2=0.014)
    }
    if dist_type in func_switcher.keys():
        # https://books.google.com/books?id=OxlBqY67rl0C&pg=PA31&vq=1.42&dq=jnd+gaurav+sharma#v=onepage&q=1.42&f=false
        # https://en.wikipedia.org/wiki/CIELAB_color_space
        # https://en.wikipedia.org/wiki/Just-noticeable_difference
        # JND = 2.3 if CIE76
        lab_a = convert_color(rgb_a, LabColor)
        lab_b = convert_color(rgb_b, LabColor)
        delta_e_func = func_switcher.get(dist_type)
        dist = delta_e_func(lab_a, lab_b)
        return dist
    # Euclidean with sRGB as default
    # https://www.compuphase.com/cmetric.htm
    avg_r = (rgb_a.rgb_r + rgb_b.rgb_r) * 255 / 2
    w_r = 2 + avg_r / 256
    delta_r = (rgb_a.rgb_r - rgb_b.rgb_r) * 255
    w_g = 4
    delta_g = (rgb_a.rgb_g - rgb_b.rgb_g) * 255
    w_b = 2 + (255 - avg_r) / 256
    delta_b = (rgb_a.rgb_b - rgb_b.rgb_b) * 255
    dist = math.sqrt(w_r * pow(delta_r, 2) + w_g * pow(delta_g, 2) + w_b * pow(delta_b, 2))
    return dist


def find_nearest_color(input_rgb: tuple, dist_type: str = 'sRGB'):
    hex_code_a = rgb_to_hex(input_rgb)
    rgb_a = sRGBColor(*input_rgb, is_upscaled=True)
    nearest_color = None
    for _, each_color in df_color_dict.iterrows():
        hex_code_b = each_color['hex_code']
        tuple_rgb_b = hex_to_rgb(hex_code_b)
        color_names = re.split('[,]', each_color['color_names_zh'])
        rgb_b = sRGBColor(*tuple_rgb_b, is_upscaled=True)
        dist = color_distance(rgb_a, rgb_b, dist_type=dist_type)
        if nearest_color is None or nearest_color.get('distance') > dist:
            nearest_color = {
                'hex_code_origin': hex_code_a,
                'hex_code': hex_code_b,
                'color_names': color_names,
                'distance': dist
            }
    return nearest_color


if __name__ == '__main__':
    input_rgb = (204, 163, 100)
    # https://www.colortell.com/colorspeed
    print(find_nearest_color(input_rgb, dist_type='sRGB'))
    print(find_nearest_color(input_rgb, dist_type='CIE76'))
    print(find_nearest_color(input_rgb, dist_type='CIE94'))

    input_rgb = (240, 230, 230)
    print(find_nearest_color(input_rgb, dist_type='sRGB'))
    print(find_nearest_color(input_rgb, dist_type='CIE76'))
    print(find_nearest_color(input_rgb, dist_type='CIE94'))

    input_rgb = (245, 246, 251)
    print(find_nearest_color(input_rgb, dist_type='sRGB'))
    print(find_nearest_color(input_rgb, dist_type='CIE76'))
    print(find_nearest_color(input_rgb, dist_type='CIE94'))

    input_rgb = (230, 223, 226)
    print(find_nearest_color(input_rgb, dist_type='sRGB'))
    print(find_nearest_color(input_rgb, dist_type='CIE76'))
    print(find_nearest_color(input_rgb, dist_type='CIE94'))
