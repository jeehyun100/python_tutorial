# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2
# matplot이 안보이면 아래 커맨드를 실행
# plt.interactive(True)


def read_img(file_name, read_type='plt'):
    """
    File name으로 그림파일을 읽는 기능
    read type으로 matplot으로 읽을지 opencv로 읽을지 결정한다.

    Args:
        file_name, String: 카드의 모양, ['Hearts', 'Clubs', 'Diamonds', 'Spades']
        read_type, plt or not plt

    return:
        numpy.array: The image data
    """
    if read_type == 'plt':
        _buf = plt.imread(file_name)
    else:
        _buf = cv2.imread(file_name, 1)
        _buf = cv2.cvtColor(_buf, cv2.COLOR_BGR2RGB)
    return _buf


def split_channel(buf_arr):
    """
    채널을 분리하고, 각 채널별로 최대값과 평균값을 구하는 기능

    Args:
        buf_arr, numpy.array: The image data

    return:
        tuple, tuple, tuple: (Red channel array, max value, mean value), Green tuple, Blue tuple
    """
    # 채널을 분리할 numpy array 생성
    _tmp_r = np.zeros(buf_arr.shape, dtype=buf_arr.dtype)
    _tmp_g = np.zeros(buf_arr.shape, dtype=buf_arr.dtype)
    _tmp_b = np.zeros(buf_arr.shape, dtype=buf_arr.dtype)

    # 각 채널별로 복사
    _tmp_r[:, :, 0] = buf_arr[:, :, 0]
    _tmp_g[:, :, 1] = buf_arr[:, :, 1]
    _tmp_b[:, :, 2] = buf_arr[:, :, 2]

    # matplot은 값이 normalize되어 있어서 255를 곱해서 원래 값으로 복원
    if _tmp_r.dtype != np.uint8:
        _tmp_r = (255*_tmp_r).astype(np.uint8)
    if _tmp_g.dtype != np.uint8:
        _tmp_g = (255*_tmp_g).astype(np.uint8)
    if _tmp_b.dtype != np.uint8:
        _tmp_b = (255*_tmp_b).astype(np.uint8)

    # 각채널별로 max와 mean을 구함
    _r_max = _tmp_r[:, :, 0].max()
    _r_arg = _tmp_r[:, :, 0].mean()
    _g_max = _tmp_g[:, :, 1].max()
    _g_arg = _tmp_g[:, :, 1].mean()
    _b_max = _tmp_b[:, :,  2].max()
    _b_arg = _tmp_b[:, :, 2].mean()

    return (_tmp_r, _r_max, _r_arg), (_tmp_g, _g_max, _g_arg), (_tmp_b, _b_max, _b_arg)


def change_channel(buf_arr):
    """
    GBR로 채널을 변경하는 기능

    Args:
        buf_arr, numpy.array: The image data

    return:
        buf_arr, numpy.array: The converted image data (G, B, R)
    """
    _buf_bgr = np.zeros(buf_arr.shape, dtype=buf_arr.dtype)
    _buf_bgr[:, :, 0] = buf_arr[:, :, 1]
    _buf_bgr[:, :, 1] = buf_arr[:, :, 2]
    _buf_bgr[:, :, 2] = buf_arr[:, :, 0]
    return _buf_bgr


def plot_with_anno_helper(figure, buf_arr, index, arr_max=0, arr_arg=0, channel='A'):
    """
    문제 2번에 맞게 plotting 을 도와주는 기능

    Args:
        figure, matplotlib.figure :
        buf_arr, numpy.array : The image data
        index, Integer : sub plot index
        arr_max, Float : Max value of the image data
        arr_arg, Float : Mean value of the imege data
        channel, String : Channel name 예) R, G, G

    """
    ax2 = figure.add_subplot(3, 2, index)
    ax2.imshow(buf_arr)
    # 인덱스 별로 subplot의 Title을 그리기 위한 list
    titles = ['Original', 'Red', 'Green', 'Blue', 'Convert GBR']
    ax2.title.set_text(titles[index-1])
    # x,y축은 표시 안함
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Original 과 convert GBR은 주석을 표시 안함
    if channel != 'A':
        bbox_props_f = dict(boxstyle='round', fc='w', lw=2)
        r_max_anno = "{0}_max : {1}".format(channel, arr_max)
        r_arg_anno = "{0}_arg : {1}".format(channel, round(arr_arg, 2))
        ax2.annotate(r_max_anno, xy=(50, 70), bbox=bbox_props_f)
        ax2.annotate(r_arg_anno, xy=(50, 160), bbox=bbox_props_f)


if __name__ == "__main__":
    # 전체 plot을 설정한다. 이름을 Problem 2로 한다.
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle('Problem 2')

    # 문제 2.(가)의 답
    # q1.png를 읽고 plotting 한다.
    buf = read_img('q1.png', 'plt')
    plot_with_anno_helper(fig, buf, 1)

    # 문제 2.(나)의 답
    (r_buf, r_max, r_arg), (g_buf, g_max, g_arg), (b_buf, b_max, b_arg) = split_channel(buf)
    plot_with_anno_helper(fig, r_buf, 2, r_max, r_arg, "R")
    plot_with_anno_helper(fig, g_buf, 3, g_max, g_arg, "G")
    plot_with_anno_helper(fig, b_buf, 4, b_max, b_arg, "B")

    # 문제 2.(다)의 답
    change_buf = change_channel(buf)
    plot_with_anno_helper(fig, change_buf, 5)
    plt.show()
