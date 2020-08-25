from colorthief import ColorThief
from img2tags.colornames import find


def extract_main_color(fp):
    color_thief = ColorThief(fp)
    # build a color palette
    palette = color_thief.get_palette(color_count=5)
    return [find(each_color) for each_color in palette]
