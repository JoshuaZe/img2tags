from colorthief import ColorThief
from color.colornames import find

if __name__ == '__main__':
    # https://github.com/fengsp/color-thief-py
    # https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
    # https://www.hisour.com/zh/color-appearance-model-24824/
    color_thief = ColorThief('/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000001_855.jpg')
    # build a color palette
    palette = color_thief.get_palette(color_count=5)
    print(palette)

    for color in palette:
        result = find(color)
        print(result)

