import math
import torch
from torchvision import transforms as transforms

# saliency
import csv
import numpy as np
from pyquaternion import Quaternion
from utils import get_fixation
from matplotlib import animation
import matplotlib.pyplot as plt

transform1 = transforms.Compose([
    transforms.ToTensor(),  # 归一化操作range [0, 255] -> [0.0,1.0]
])
transform2 = transforms.Compose([
    transforms.ToTensor(),  # 归一化 --> [-1.0, 1.0]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# 图片转化成Tensor
def imageofCVToTensor(imgdata):
    img1 = imgdata[0]
    imgT1 = transform1(img1)
    img2 = imgdata[1]
    imgT2 = transform1(img2)
    TenTemp1 = torch.stack((imgT1, imgT2), 0)
    for i in range(2, len(imgdata), 2):
        img1 = imgdata[i]
        imgT1 = transform1(img1)
        img2 = imgdata[i + 1]
        imgT2 = transform1(img2)
        TenTemp2 = torch.stack((imgT1, imgT2), 0)
        TenTemp1 = torch.cat((TenTemp1, TenTemp2), 0)
    return TenTemp1


# 计算用户视野范围
def CalculateUserView(x, y, W_Frame, H_Frame, W_Tile, H_Tile):
    """
    return:
        UWL UHL 应该是矩形的左上角坐标
        UWH UHH 应该是矩形的右下角坐标
        W, H 表示动态设定的用户视野范围 --> 可以这样吗？
     """
    W = int(W_Tile * 0.5)  # 注意这里需要传入整数
    H = int(H_Tile * 0.75)
    UWL = x - W
    UHL = y - H
    UWH = x + W
    UHH = y + H

    if UWL < 0:
        UWL = 0
    if UHL < 0:
        UHL = 0
    if UWH > W_Frame:
        UWH = int(W_Frame)
    if UHH > H_Frame:
        UHH = int(H_Frame)

    return UWL, UHL, UWH, UHH


# 检测预测结果？
def CheckPredictResult(TileForCheck, UWL, UHL, UWH, UHH, W_Tile, H_Tile):
    """
    UWL, UHL 表示用户矩形视野左上角坐标 --> (iL, jL)
    UWH, UHH 表示用户矩形视野右上角坐标 --> (iH, jH)
    计算矩形四个角的坐标，形如：
        (iL, jL) ------ (iH, jL)
            |                |
            |                |
            |                |
        (iL, jH) ------ (iH, jH)
    return :
        Flag(acc for each frame): 每一帧的用户视野是否被预测的tiles所覆盖
        T(countMatched): （0~4）in 25 tiles
     """
    iL = int(math.floor(UWL / W_Tile))
    jL = int(math.floor(UHL / H_Tile))
    iH = int(math.floor(UWH / W_Tile))
    jH = int(math.floor(UHH / H_Tile))
    Flag = 0
    if iH >= 5:
        iH = 4
    if jH >= 5:
        jH = 4
    try:
        A = TileForCheck[jL * 5 + iL]
        B = TileForCheck[jL * 5 + iH]
        C = TileForCheck[jH * 5 + iL]
        D = TileForCheck[jH * 5 + iH]
    except:
        print("CheckList out of index")
        exit()
    T = A + B + C + D
    if A * B * C * D == 1:
        Flag = 1
    # if T >= 3:
    #     Flag = 1
    return Flag, T


def CheckPredictResFeedback(TileForCheck, TileByFeedback, UWL, UHL, UWH, UHH, W_Tile, H_Tile):
    iL = int(math.floor(UWL / W_Tile))
    jL = int(math.floor(UHL / H_Tile))
    iH = int(math.floor(UWH / W_Tile))
    jH = int(math.floor(UHH / H_Tile))
    Flag = 0
    if iH >= 5:
        iH = 4
    if jH >= 5:
        jH = 4
    try:
        A = TileByFeedback[jL * 5 + iL]
        B = TileByFeedback[jL * 5 + iH]
        C = TileByFeedback[jH * 5 + iL]
        D = TileByFeedback[jH * 5 + iH]
    except:
        print("FeedbackList out of index")
        exit()
    T = A + B + C + D
    if A * B * C * D == 1:
        Flag = 1
    CountModify = 0
    for i in range(len(TileForCheck)):
        if TileForCheck[i] == 0 and TileByFeedback[i] == 1:
            TileForCheck[i] = 1
            CountModify += 1
    return Flag, CountModify


# 有点类似于LSTM的Feedback
def UpdateTiletFeedback(TileByFeedback, UWL, UHL, UWH, UHH, W_Tile, H_Tile):
    for i in range(len(TileByFeedback)):
        TileByFeedback[i] = 0
    iL = int(math.floor(UWL / W_Tile))
    jL = int(math.floor(UHL / H_Tile))
    iH = int(math.floor(UWH / W_Tile))
    jH = int(math.floor(UHH / H_Tile))
    Flag = 0
    if iH >= 5:
        iH = 4
    if jH >= 5:
        jH = 4
    for i in range(iL, iH + 1):
        for j in range(jL, jH + 1):
            TileByFeedback[j * 5 + i] = 1


# saliency相关的函数
def de_interpolate(raw_tensor, N, H, W, tileNo):
    """
    F.interpolate(source, scale_factor=scale, mode="nearest")的逆操作！
    :param raw_tensor: [B, C, H, W]
    :return: [B, C, H // 2, W // 2]
    """
    out = np.zeros((N, tileNo, tileNo))

    H_interval = math.floor(H / tileNo)
    W_interval = math.floor(W / tileNo)
    for i in range(H_interval):
        for j in range(W_interval):
            out = out + raw_tensor[:, i::H_interval, j::W_interval]
    out = out / out.sum(axis=2).sum(axis=1).mean()
    return out


def data_prepare(idx, idy, UserId, N):
    """
    idx: 实验Id
    idy: VideoId
    UserId: 用户Id
    N: 头部轨迹记录数
    """
    Userdata = []
    UserFile = 'D:/Multimedia/FoV_Prediction/Dataset/VRdataset/Experiment_' + str(idx) + '/' + str(
        UserId) + "/video_" + str(idy) + ".csv"

    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csv file中的文件
        csvLength = np.array(list(csv_reader)).shape[0]

    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csv file中的文件
        birth_header = next(csv_reader)
        index = 0
        for row in csv_reader:
            if index % (csvLength / N) < 1:
                v0 = [0, 0, 1]
                q = Quaternion([float(row[4]), -float(row[3]), float(row[2]), -float(row[1])])
                Userdata.append(q.rotate(v0))

            index = index + 1

    Userdata = np.array(Userdata)
    return Userdata


def vector_to_ang(_v):
    # v = np.array(vector_ds[0][600][1])
    # v = np.array([0, 0, 1])
    _v = np.array(_v)
    alpha = get_fixation.degree_distance(_v, [0, 1, 0])  # degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [0, np.cos(alpha / 180.0 * np.pi), 0]  # proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = _v - proj1  # proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = get_fixation.degree_distance(proj2,
                                         [1, 0, 0])  # theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if get_fixation.degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi


def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h / 2.0 - (_h / 2.0) * np.sin(_phi / 180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0 / 360 * _w)
    return int(x), int(y)


def create_fixation_map(_X, _y, _idx, H, W):
    v = _y[_idx]
    theta, phi = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H - hi - 1, W - wi - 1] = 1
    return result


def display_frames_as_gif(policy, frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./' + policy + '_viewport_result.gif', writer='ffmpeg', fps=30)
