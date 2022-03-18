from pandas import DataFrame
from pandas import concat
import cv2 as cv
import numpy as np
import csv
import math


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    将时间序列重构为监督学习数据集.
    参数:
        data: 观测值序列，类型为列表或Numpy数组。
        n_in: 输入的滞后观测值(X)长度。
        n_out: 输出观测值(y)的长度。
        dropnan: 是否丢弃含有NaN值的行，类型为布尔值。
    返回值:
        经过重组后的Pandas DataFrame序列.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 将列名和数据拼接在一起
    agg = concat(cols, axis=1)
    agg.columns = names
    # 丢弃含有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 根据四元数计算位置
def LocationCalculate(x, y, z, w):
    X = 2 * x * z + 2 * y * w
    Y = 2 * y * z - 2 * x * w
    Z = 1 - 2 * x * x - 2 * y * y

    a = np.arccos(np.sqrt(X ** 2 + Z ** 2) / np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
    if Y > 0:
        ver = a / np.pi * 180
    else:
        ver = -a / np.pi * 180

    b = np.arccos(X / np.sqrt(X ** 2 + Z ** 2))
    if Z < 0:
        hor = b / np.pi * 180
    else:
        hor = (2. - b / np.pi) * 180

    return (90 - ver) / 180, hor / 360


# 计算视频每1s内用户观看的位置: 也就是每帧的用户观看位置
def userLocal_One(FrameRate, UserFile, TotalSeconds, FH, FW):
    """ 
    :param FrameRate: FPS
    :param UserFile: The filename of file which stores user's viewport location data
    :param TotalFrames: 帧总数
    :param FH and FW: 帧宽度和帧高度
    :return: UserLocationPerFrame: length = TotalSeconds * FrameRate
    """

    """
        Read user data file and collect all records
        Save them in two lists: Userdata and TimeStamp
        Userdata is where store the user location (convert to float)
        TimeStamp is for syncronization
    """

    Userdata = []
    flagTime = 1
    TimeStamp = []
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到Userdata中
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])
    flagTime = 0
    Userdata = [[float(x) for x in row] for row in Userdata]  # 将数据从string形式转换为float
    # Userdata = np.array(Userdata)  # 将list数组转化成array数组便于查看数据结构
    # print("len of Userdata[0]:", len(Userdata[0]))
    strItem = TimeStamp[0].split(':')
    PreTime = math.ceil(float(strItem[2]))
    CurTime = math.floor(float(strItem[2]))
    NumberCount = 0
    UserLocationPerFrame = []
    all_dataPerFrame = []
    j = 0  # j for all items in one user
    while NumberCount < TotalSeconds:
        NumberCount += 1
        UserAll = []
        Datain1s = []
        '''
        通过用户数据中第一列，时间戳信息，获取每一秒内的用户location。该秒内用户数据记录可能大于帧率也可能小于帧率
        '''
        while PreTime > CurTime:
            # TimeStamp[j] 形如 2016-11-17 02:14:34.804
            if (j >= len(TimeStamp)):
                print('UserFileError...Out of index for TimeStamp. len(UserLocationPerFrame)=', len(UserLocationPerFrame))
                exit()
            strA = TimeStamp[j].split(':')
            CurTime = math.floor(float(strA[2]))
            if CurTime == 0 and PreTime == 60:
                break
            x = Userdata[j][1]
            y = Userdata[j][2]
            z = Userdata[j][3]
            w = Userdata[j][4]
            H, W = LocationCalculate(x, y, z, w)
            IH = math.floor(H * FH)
            IW = math.floor(W * FW)
            UserAll.append([IW, IH])    # 应该是计算用户视野范围
            Datain1s.append(Userdata[j][1:] + [IW, IH])
            # print(">>>>>>>>>>>>>>>>     ", H,W, "   <<<<<<<<<<<<<<<<<<<<<<<")
            # print(IW,IH)
            j = j + 1
        PreTime = CurTime + 1
        '''
        获得每一秒内用户视角记录后，整理每一帧用户视角位置
        '''
        LengthInOneSec = len(UserAll)
        if LengthInOneSec >= FrameRate:
            IntervalIndex = LengthInOneSec / FrameRate
            for IU in range(FrameRate):
                ModiIndex = int(round(IU * IntervalIndex))
                if ModiIndex >= len(UserAll):
                    print("Large than FrameRate", ModiIndex, len(UserAll))
                UserLocationPerFrame.append(UserAll[ModiIndex])
                all_dataPerFrame.append(Datain1s[ModiIndex])
        else:
            IntervalIndex = LengthInOneSec / FrameRate
            for IU in range(FrameRate):
                ModiIndex = int(round(IU * IntervalIndex))
                if ModiIndex >= len(UserAll):
                    print("Less than FrameRate", ModiIndex, len(UserAll))
                    ModiIndex = len(UserAll) - 1
                UserLocationPerFrame.append(UserAll[ModiIndex])
                all_dataPerFrame.append(Datain1s[ModiIndex])
    return UserLocationPerFrame, all_dataPerFrame


# 预测是否正确
def IfPInUserView(x, y, xp, yp, W_Frame, H_Frame):
    """ 
    x, y 表示用户实际观看视野，以为(x, y)为中心的边长为400的正方形区域
        (UWL, UHL) ------ (UWH, UHL)
            |                |
            |                |
            |                |
        (UWL, UHH) ------ (UWH, UHH)
     """
    W = 200
    H = 200
    UWL = x - W
    UHL = y - H
    UWH = x + W
    UHH = y + H

    if UWL < 0:
        UWL = 0
    if UHL < 0:
        UHL = 0
    if UWH > W_Frame:
        UWH = W_Frame
    if UHH > H_Frame:
        UHH = H_Frame
    flag = 0
    if UWL <= xp and xp <= UWH and UHL <= yp and UHH >= yp:
        flag = 1
    return flag



# normal 
def processed_data_200_Tiles(W_Frame, H_Frame, grid_size=3, index = 1, u = [] , f = []):
    """ 
    :param grid_size: 表示TileNumber 5 * 5的分块大小
    :param index: 表示第几个tile
    :param u: 表示用户观看frame时的位置 与f对应 len(u) = 8
    :param f: 表示用户观看frame 与u对应 是一个三维张量 len(f) = 8
    :param W_Frame and H_Frame: the width and height of frame
    return : 
        p_u 表示用户视野是否在tile内 只有0，1 和p_f对应
        p_f 存放每个tile 和p_u对应
        Uview 存放在视野范围内的tile 对应于p_u中1的部分
     """
    # u, f = get_data()
    p_u, p_f = [], []
    Uview = []

    # print("数据格式为",f[0].shape,"视频图片长度为：",len(f),"用户数据长度为：",len(u))
    # print("用户数据为：",u)

    for i in range(len(u)):
        frame = f[i]
        row = math.ceil(index / grid_size)
        col = (index - 1) % grid_size
        """ 某一个tile的四个角的坐标: (x: col, y: row)
            (xp, yp) ------ (xph, yp)
                |                |
                |                |
                |                |
            (xp, yph) ------ (xph, yph)
         """
        yp = (H_Frame / grid_size) * (row - 1)
        xp = (W_Frame / grid_size) * (col)
        yph = (H_Frame / grid_size) * (row)
        xph = (W_Frame / grid_size) * (col + 1)
        """ 
        frame是一个三维张量，<class 'numpy.ndarray'>(height, width, RGB) (H_Frame, W_Frame, 3)
         """
        try:
            frame_part = frame[int(yp):int(yph), int(xp):int(xph), :]
        except:
            print("ERROR for frame_part: ", frame.shape, yp, yph, xp, xph)
            
        frame_part = cv.resize(frame_part, (32, 32))      # 120*120
        p_f.append(frame_part)
        
        pos = u[i]  # 获取观看某帧时用户的视野点坐标
        x = pos[0]
        y = pos[1]
        A = IfPInUserView(x, y, xp, yp, W_Frame, H_Frame)
        B = IfPInUserView(x, y, xp, yph, W_Frame, H_Frame)
        C = IfPInUserView(x, y, xph, yp, W_Frame, H_Frame)
        D = IfPInUserView(x, y, xph, yph, W_Frame, H_Frame)

        if A==1 or B==1 or C==1 or D==1:
            p_u.append(1)
            Uview.append(frame[int(yp):int(yph), int(xp):int(xph), :])
        else:
            p_u.append(0)

    # p_u = np.asarray(p_u)
    # p_f = np.asarray(p_f)
    # print("整理后数据格式",  p_f.shape, "整理后图片数目为：", len(p_f), "用户数据长度为：", len(p_u),"用户数据格式为：",p_u.shape)

    return p_u, p_f, Uview


# normal
def processed_data_200_Tiles_all(W_Frame, H_Frame, grid_size=3, index=1, u=[]):
    """
    :param grid_size: 表示TileNumber 5 * 5的分块大小
    :param index: 表示第几个tile
    :param u: 表示用户观看frame时的位置 与f对应 len(u) = 8
    :param W_Frame and H_Frame: the width and height of frame
    return :
        p_u 表示用户视野是否在tile内 只有0，1 和p_f对应
        p_f 存放每个tile 和p_u对应
        Uview 存放在视野范围内的tile 对应于p_u中1的部分
     """
    p_u = []

    for i in range(len(u)):
        row = math.ceil(index / grid_size)
        col = (index - 1) % grid_size
        """ 某一个tile的四个角的坐标: (x: col, y: row)
            (xp, yp) ------ (xph, yp)
                |                |
                |                |
                |                |
            (xp, yph) ------ (xph, yph)
         """
        yp = (H_Frame / grid_size) * (row - 1)
        xp = (W_Frame / grid_size) * (col)
        yph = (H_Frame / grid_size) * (row)
        xph = (W_Frame / grid_size) * (col + 1)

        pos = u[i]  # 获取观看某帧时用户的视野点坐标
        x = pos[0]
        y = pos[1]
        A = IfPInUserView(x, y, xp, yp, W_Frame, H_Frame)
        B = IfPInUserView(x, y, xp, yph, W_Frame, H_Frame)
        C = IfPInUserView(x, y, xph, yp, W_Frame, H_Frame)
        D = IfPInUserView(x, y, xph, yph, W_Frame, H_Frame)

        if A == 1 or B == 1 or C == 1 or D == 1:
            p_u.append(1)
        else:
            p_u.append(0)

    return p_u
