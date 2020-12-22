#!/usr/bin/env python
# 这是我实现的OFD，但是“去闪烁”效果一般。分析可能是只有在抠图这样单通道的任务上比较适用，彩色图去抖动则要复杂得多。
import numpy as np
from PIL import Image
from tqdm import tqdm
import os


def OFD(video_ofd_i, video_ofd_o):
    """One-Frame Delay
    Note that OFD is only suitable for smooth movement. It may fail in fast motion videos.
    """
    α = 240

    img_list = os.listdir(video_ofd_i)
    img_list.sort(key=lambda x: int(x[:-4]))

    img_path_first = os.path.join(video_ofd_i, img_list[0])
    img_path_last = os.path.join(video_ofd_i, img_list[-1])
    os.system(f"cp {img_path_first} {video_ofd_o}/{img_list[0]}")
    os.system(f"cp {img_path_last} {video_ofd_o}/{img_list[-1]}")

    for index in tqdm(range(1, len(img_list)-1)):
        list_mf = []
        for i in [index - 1, index, index + 1]:

            if not img_list[i].endswith('png'):
                continue

            img_path = os.path.join(video_ofd_i, img_list[i])
            img = Image.open(img_path).convert('RGB')
            img = np.asarray(img)
            list_mf.append(img)

        t02 = np.abs(list_mf[0] - list_mf[2]).astype(np.float32)
        t01 = np.abs(list_mf[0] - list_mf[1]).astype(np.float32)
        t12 = np.abs(list_mf[1] - list_mf[2]).astype(np.float32)

        m02 = (list_mf[0] + list_mf[2]) / 2.0

        C = np.where((t02 <= α) & (t01 >= α) & (t12 >= α), 1, 0)
        print('flickering pixels', C.sum())

        out = np.where(C == 1, m02, list_mf[1])

        out = out.clip(0, 255)
        out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')
        output_path = os.path.join(video_ofd_o, img_list[index])
        out_img.save(output_path)

if __name__=='__main__':
    print('===> One-Frame Delay')
    OFD("./INPUT", "./OFD")
