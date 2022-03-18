# added arguments.py
import time
import torch.backends.cudnn as cudnn
from torch import nn, optim

import cv2 as cv
from model import *

from utils import video_process as vProcessor
from utils.out_writer import allWriter
from utils.utils import *
from utils.video_process import processed_data_200_Tiles_all


class LiveVideo(object):

    def __init__(self, config):
        self.userId = config.userId
        self.videoId = config.videoId

        self.video_path = config.videos_path
        self.records_path = config.records_path
        self.log_path = config.log_path
        self.model_path = config.model_path + 'CNN/modelCNN.pth'
        self.sampleRate = config.sampleRate
        self.modelCNN = None
        self.visId = config.visId
        self.epoch = config.epochGCN
        self.criterionCNN = None
        self.optimizerCNN = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.totalFrames = 0

        fileList = [" ", "1-1-Conan Gore Fly", "1-2-Front", "1-3-Help", "1-4-Conan Weird Al",
                    "1-5-Tahiti Surf", "1-6-Falluja", "1-7-Cooking Battle", "1-8-Football",
                    "1-9-Rhinos", "2-1-Korean", "2-2-VoiceToy", "2-3-RioVR", "2-4-FemaleBasketball",
                    "2-5-Fighting", "2-6-Anitta", "2-7-TFBoy", "2-8-Reloaded"]

        self.videoName = fileList[self.videoId]
        # 视频路径
        self.videoFile = self.video_path + self.videoName + ".mp4"

        self.allWriter = allWriter(config, self.videoName)

        self.tileNO = 5  # TileNo: 分块大小 (5 * 5)
        self.totalTile = self.tileNO * self.tileNO  # 25 tiles
        self.countMain = 0  # 程序主计数器
        self.colorPrediction = (255, 0, 0)  # BGR
        self.baseId = -1  # 定位LocationFrame的基数

        self.thres_factor = -1

        # for CNN

        # for RL

        # END

    def load_CNN_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.modelCNN = GCN().to(self.device)

        """ 
        选择不同的优化函数计算梯度
        """
        self.optimizerCNN = optim.Adam(self.modelCNN.parameters(), lr=self.lrCNN)
        self.schedulerCNN = optim.lr_scheduler.MultiStepLR(self.optimizerCNN, milestones=[75, 150], gamma=0.5)
        """ 
        计算Loss应该根据模型来选择
        """
        self.criterionCNN = nn.CrossEntropyLoss().to(self.device)

    def load_sali_model(self):

        if os.access(self.model_path, os.X_OK):
            checkpoint = torch.load(self.model_path)
            self.modelCNN.load_state_dict(checkpoint['net'])
            self.optimizerCNN.load_state_dict(checkpoint['optimizer'])
            self.criterionCNN.load_state_dict(checkpoint['criterion'])
        else:
            stateCNN = {
                'net': self.modelCNN.state_dict(),
                'optimizer': self.optimizerCNN.state_dict(),
                'criterion': self.criterionCNN.state_dict()
            }
            torch.save(stateCNN, self.model_path)

    def save_sali_model(self):
        stateCNN = {
            'net': self.modelCNN.state_dict(),
            'optimizer': self.optimizerCNN.state_dict(),
            'criterion': self.criterionCNN.state_dict()
        }
        torch.save(stateCNN, self.model_path)

    # 显著性特征的在线训练函数
    def trainCNN(self, output, sds, totalT1):

        target, TileForCheck = self.testCNN(output, sds)

        """
        CNN在线训练，提取显著性特征
        """
        startT2 = time.time()
        for Times in range(self.epoch):
            self.optimizerCNN.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0

            if Times != 0:
                output = self.modelCNN(self.TenFram)

            loss = self.criterionCNN(output, target)

            if Times == 0:
                FirstLoss = loss.item()

            loss.backward()

            self.optimizerCNN.step()

            # 调整学习率
            if self.schedulerCNN is not None:
                self.schedulerCNN.step()

        endT2 = time.time()
        ToTalTime = endT2 - startT2 + totalT1
        print(f"==========Time of training for {self.baseId} segment(200 tiles): {ToTalTime}")

        FinalLoss = loss.item()
        print(
            "baseId:", self.baseId,
            "Epoch:", self.epoch,
            "loss: %.4f --> %.4f" % (FirstLoss, FinalLoss)
        )

        self.allWriter.writerCSVT([
            str(ToTalTime),
            str(int(self.epoch)),
            str(float(FirstLoss)),
            str(float(FinalLoss))
        ])

        return TileForCheck

    # 显著性特征的在线训练函数
    def testCNN(self, output, sds):
        """
        New updated result output
        """
        predictionCNN = []
        realSali = []
        inter = int(self.bufLen / self.sampleRate)
        for i in range(inter):
            for j in range(self.totalTile):
                predictionCNN.append(output[j * inter + i].data)
                realSali.append(sds[j * inter + i])

        # 记录各项指标，以每8帧为单位记录
        TP, TN, FP, FN = 0, 0, 0, 0
        PredictedTile = 0
        for j in range(inter):
            Tile_Status = []
            Ground_Truth = []
            for i in range(self.totalTile):
                Tile_Status.append(predictionCNN[j * self.totalTile + i][0])
                Ground_Truth.append(realSali[j * self.totalTile + i])

            SortTile_Status = Tile_Status.copy()
            SortTile_Status.sort()
            indexTmp = int(self.totalTile / 2 + (self.totalTile / 8) * self.thres_factor)
            indexTmp = min([max([0, indexTmp]), 24])
            Thres = SortTile_Status[indexTmp]

            TileForCheck = []
            for k in range(self.totalTile):
                if Tile_Status[k] < Thres:
                    TileForCheck.append(1)
                    PredictedTile += 1
                else:
                    TileForCheck.append(0)

            for k in range(self.totalTile):
                if TileForCheck[k] + Ground_Truth[k] == 2:
                    TP += 1
                elif TileForCheck[k] + Ground_Truth[k] == 0:
                    TN += 1
                elif TileForCheck[k] == 0 and Ground_Truth[k] == 1:
                    FP += 1
                else:
                    FN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        if (TP + FP) == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if (TP + FN) == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        if precision >= 0.8 and recall < 0.6:
            self.thres_factor += -0.5
        elif precision < 0.8:
            self.thres_factor += +0.5

        avePreTile = PredictedTile / inter

        sortedValues = [str(accuracy), str(precision), str(recall), str(avePreTile), str(self.totalTile)]
        self.allWriter.writerCSVA(sortedValues)
        print("node ID:", str(self.userId),
              "accuracy:", str(accuracy),
              "precision:", str(precision),
              "recall", str(recall),
              "predicted tile", str(avePreTile))

        user = torch.from_numpy(np.array(sds))
        target = user.long().to(self.device)  # torch.Size([200])

        return target, TileForCheck

    # 加载视频数据
    def videoLoad(self):
        tmp1 = self.videoName[0]  # 视频组 [1, 2]
        tmp2 = int(self.videoName[2]) - 1  # 视频序号 1.[1-9] 2.[1-8]

        self.cap = cv.VideoCapture(self.videoFile)
        self.capB = cv.VideoCapture(self.videoFile)

        '''
        用户数据
        '''
        # 用户观看行为路径
        userRecords = self.records_path + tmp1 + '/' + str(self.userId) + "/video_" + str(tmp2) + ".csv"
        userDataCSV = userRecords

        '''
        视频信息
        '''
        self.W_Frame = self.cap.get(3)
        self.H_Frame = self.cap.get(4)
        #
        print("===============" + self.videoName + "============")
        print("Frame width:", self.W_Frame)
        print("Frame height:", self.H_Frame)
        self.frameRate = int(round(self.cap.get(5)))  # 获取每秒帧率

        self.totalFrames = self.cap.get(7)  # 视频总帧数
        self.TotalSeconds = int(round(self.totalFrames / self.frameRate))  # 视频时长
        print("Frame rate:", self.frameRate)
        print("Total frames:", self.totalFrames)
        print("Video total second (totalFrames / framerate):", self.TotalSeconds)

        # 这里只是处理一个用户吗，需要修改成处理多个用户
        self.LocationPerFrame, self.all_dataPF = vProcessor.userLocal_One(self.frameRate, userDataCSV,
                                                                          self.TotalSeconds, self.H_Frame, self.W_Frame)

        print("Total frame from user data ==> len(LocationPerFrame)", len(self.LocationPerFrame))
        self.interFOV = len(self.LocationPerFrame) / self.totalFrames

        self.W_Tile, self.H_Tile = self.W_Frame / self.tileNO, self.H_Frame / self.tileNO  # W_Tile H_Tile: 块的宽度和高度
        bufInSecond = 8  # bufInSecond: buffer长度bufLen的基数
        self.bufLen = self.sampleRate * bufInSecond  # bufLen: 30 * 2 = 60

        # 生成log的CSV文件
        self.allWriter.writerHead(self.tileNO)
        # 生成视频文件
        self.out = self.allWriter.writerVideo(self.frameRate, self.W_Frame, self.H_Frame)

    def get_time(self):
        return int(self.totalFrames / self.bufLen)

    # 将viewpoint转化为200维向量
    def stateToVec(self, next_state):
        state_vec = np.zeros([200])
        add_one = next_state
        realSali = []
        for j in range(25):
            au = processed_data_200_Tiles_all(self.W_Frame,
                                              self.H_Frame,
                                              self.tileNO,
                                              j + 1,
                                              add_one)
            if len(au) == 0:
                return realSali
            state_vec[8 * j:8 * (j + 1)] = au

        for i in range(8):
            for j in range(25):
                realSali.append(state_vec[j * 8 + i])  # 检查视野是否输出正确？行列是否正确？

        return realSali

    def get_history(self):
        # 获取观看者历史观看点
        view_point = self.getNextSate()

        # 获取用户的实际观看记录
        view_point_fix = []
        for index, value in enumerate(view_point):
            view_point_fix.append([value[0] / self.W_Frame, value[1] / self.H_Frame])

        # 历史观看和训练完的saliency作为状态
        next_vec = self.stateToVec(view_point)

        return next_vec, view_point_fix

    def getNextSate(self):
        next_state = []
        if self.countMain + self.bufLen < self.totalFrames:
            for i in range(self.bufLen):
                if i % self.sampleRate == 0:
                    next_state.append(self.LocationPerFrame[math.ceil(self.interFOV * (self.countMain - 1)) + 1])
                self.countMain += 1
            return next_state
        else:
            return next_state

    def get_frame_tensor(self):
        # 运行的主要函数
        # 加载显著性特征的数据
        # sds = self.load_saliency()

        p_f = []  # 表示视频帧
        p_u = []  # 视频帧对应的用户观看区域
        ret = True

        # 每次处理一个buffer。 这里确保后面有一个buffer的frames
        if self.countMain + self.bufLen < self.totalFrames:
            for i in range(self.bufLen):
                ret, frame = self.cap.read()
                """ 
                添加下采样，1秒4帧就可以 bufLen = FrameRate * bufInSecond
                SubSampleRate = 4
                SubSampleStep = math.ceil(FrameRate / SubSampleRate)
                bufLen / SubSampleStep = bufInSecond * SubSampleRate = 8
                """
                if i % self.sampleRate == 0:  # 这里目标暂时是2秒的buffer中获取8帧。
                    p_f.append(frame)  # 添加图片
                    p_u.append(self.LocationPerFrame[math.ceil(self.interFOV * (self.countMain - 1)) + 1])  # 添加对应的用户数据
                self.countMain += 1
            self.baseId += 1
        else:
            return -1

        if not ret:
            print("countMain", self.countMain, "Frame out for " + self.videoName + " or Less then 2s")
            return -1

        '''
        u 表示用户视野是否在tile内 只有0，1 和p_f对应 应该可以作为label
        f 存放每个tile 和p_u对应
        v 存放在视野范围内的tile 对应于p_u中1的部分
        '''
        u, f, v = vProcessor.processed_data_200_Tiles(self.W_Frame, self.H_Frame, self.tileNO, 1, p_u, p_f)
        for index in range(1, self.totalTile):
            au, af, av = vProcessor.processed_data_200_Tiles(self.W_Frame, self.H_Frame, self.tileNO,
                                                             index + 1, p_u, p_f)
            u.extend(au)
            f.extend(af)
            v.extend(av)

        TenFram = imageofCVToTensor(f)
        self.TenFram = TenFram.to(self.device)  # torch.Size([200, 3, 32, 32])

    def run_test_step(self, all_u):

        self.get_frame_tensor()

        self.optimizerCNN.zero_grad()

        # CNN 获得图片显著性特征预测
        output = self.modelCNN(self.TenFram)

        self.testCNN(output, all_u)

    # 运行的主要函数
    def run_step(self, all_u):

        self.get_frame_tensor()

        self.optimizerCNN.zero_grad()

        # CNN 获得图片显著性特征预测
        startT1 = time.time()
        output = self.modelCNN(self.TenFram)
        endT1 = time.time()
        totalT1 = endT1 - startT1

        saliency = self.trainCNN(output, all_u, totalT1)
        return saliency