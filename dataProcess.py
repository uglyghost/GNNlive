import numpy as np
import cv2 as cv
import csv
import math
import os
import shutil

# 计算视频每1s内用户观看的位置: 也就是每帧的用户观看位置
def userLocal_One(FrameRate, UserFile, TotalSeconds, writer):
    """
    :param FrameRate: FPS
    :param UserFile: The filename of file which stores user's viewport location data
    :param TotalSeconds: 视频总时长
    :param writer: userId
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
        csv_reader = csv.reader(csvfile)    # 使用csv.reader读取csvfile中的文件
        next(csv_reader)     # 读取第一行每一列的标题
        for row in csv_reader:              # 将csv 文件中的数据保存到Userdata中
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])

    Userdata = [[float(x) for x in row] for row in Userdata]  # 将数据从string形式转换为float
    # Userdata = np.array(Userdata)         # 将list数组转化成array数组便于查看数据结构
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
        Datain1s = []
        '''
        通过用户数据中第一列，时间戳信息，获取每一秒内的用户location。该秒内用户数据记录可能大于帧率也可能小于帧率
        '''
        while PreTime > CurTime:
            # TimeStamp[j] 形如 2016-11-17 02:14:34.804
            if j >= len(TimeStamp):
                print('UserFileError...Out of index for TimeStamp. len(UserLocationPerFrame)=', len(UserLocationPerFrame))
                exit()
            strA = TimeStamp[j].split(':')
            CurTime = math.floor(float(strA[2]))
            if CurTime == 0 and PreTime == 60:
                break
            Datain1s.append(Userdata[j][0:-3])
            j += 1
        PreTime = CurTime + 1
        writer.writerow(Userdata[j][0:-3])
        '''
        获得每一秒内用户视角记录后，整理每一帧用户视角位置
        '''
        LengthInOneSec = len(Datain1s)
        if LengthInOneSec >= FrameRate:
            IntervalIndex = LengthInOneSec / FrameRate
            for IU in range(FrameRate):
                ModiIndex = int(round(IU * IntervalIndex))
                if ModiIndex >= len(Datain1s):
                    print("Large than FrameRate", ModiIndex, len(Datain1s))
                all_dataPerFrame.append(Datain1s[ModiIndex])
        else:
            IntervalIndex = LengthInOneSec / FrameRate
            for IU in range(FrameRate):
                ModiIndex = int(round(IU * IntervalIndex))
                if ModiIndex >= len(Datain1s):
                    print("Less than FrameRate", ModiIndex, len(Datain1s))
                    ModiIndex = len(Datain1s) - 1
                all_dataPerFrame.append(Datain1s[ModiIndex])
    print(f"{userId} finish!")


savePath = "D:/Multimedia/FoV_Prediction/Dataset/frames/"
sourcePath = "D:/Multimedia/FoV_Prediction/Dataset/VRdataset/Experiment_1/"
videopath = "D:/Multimedia/FoV_Prediction/Dataset/Videos/"

'''
fileList = ["1-1-Conan Gore Fly", "1-2-Front", "1-3-360 Google Spotlight Stories_ HELP", "1-4-Conan Weird Al",
                "1-5-Tahiti Surf", "1-6-Falluja", "1-7-Cooking Battle", "1-8-Football", "1-9-Rhinos",
                "2-1-Korean", "2-2-VoiceToy", "2-3-RioVR", "2-4-FemaleBasketball", "2-5-Fighting", "2-6-Anitta",
                "2-7-TFBoy", "2-8-reloaded"]
                '''

fileList = "1-5-Tahiti Surf"


def get_frame_from_video(video_name, interval):
    """
    Args:
        video_name:输入视频名字
        interval: 保存图片的帧率间隔
    Returns:
    """

    # 保存图片的路径
    save_path = video_name.split('.mp4')[0] + '/'
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)

    # 开始读视频
    video_capture = cv.VideoCapture(video_name)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if i % interval == 0:
            # 保存图片
            j += 1
            save_name = save_path + str(i) + '.jpg'
            cv.imwrite(save_name, frame)
            print('image of %s is saved' % save_name)
        if not success:
            print('video is all read')

if __name__ == "__main__":

    userIdList = list(range(1, 49))
    for videoId in [1, 2]:
        cap = cv.VideoCapture(videopath+f"{fileList}.mp4")
        W_Frame = cap.get(3)
        H_Frame = cap.get(4)
        FrameRate = int(round(cap.get(5)))
        TotalFrames = cap.get(7)
        TotalSeconds = int(round(TotalFrames / FrameRate))

        interval = 1
        get_frame_from_video(videopath+f"{fileList}.mp4", interval)

        for userId in userIdList:
            with open(savePath+f"{userId}.csv", 'w', newline='') as f:
                logWriter = csv.writer(f, dialect='excel')
                logWriter.writerow(["PlaybackTime", "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z"])

                userFile = sourcePath + f"{userId}/video_{videoId}.csv"
                userLocal_One(FrameRate, userFile, TotalSeconds, logWriter)

        cap.release()
        cv.destroyAllWindows()