import csv
import cv2


class allWriter(object):

    def __init__(self, config, videoName):
        self.videoName = videoName
        self.userId = config.userId
        self.epochGCN = config.epochGCN
        self.log_path = config.log_path

        # CSVfilenameAcc = self.videoName + '_' + str(self.userId) + 'Tk_EpoMax' + str(self.epochGCN) \
        #                  + '_AccAndBandwidth.csv'
        # fileAcc = open(config.log_path + CSVfilenameAcc, 'w', newline='')
        self.CSVfilenameTime = self.videoName + '_' + str(self.userId) + 'Tk_EpoMax' + str(self.epochGCN) \
                               + '_TimeConsumption.csv'
        #fileTime = open(self.log_path + self.CSVfilenameTime, 'w', newline='')

        # self.writerAcc = csv.writer(fileAcc)
        #self.writerTime = csv.writer(fileTime)

        #fileTime.close()

    """
    写入数据
    """
    def writerHead(self, TileNO):
        # rows = [['VideoName', 'UserIndex', 'epochRGCN'],
        #        [self.videoName, str(self.userId), str(self.epochGCN)]]
        # self.writerAcc.writerows(rows)
        # self.writerTime.writerows(rows)

        # TotalSize = 'Size/' + str(TileNO * TileNO)
        # sortedValues = ['accuracy', 'precision', 'recall', 'predicted tile', 'total size']
        # self.writerAcc.writerow(sortedValues)
        fileTime = open(self.log_path + self.CSVfilenameTime, 'w', newline='')

        ValueTime = ['accuracy', 'precision', 'recall', 'predicted tile', 'total size']
        self.writerTime = csv.writer(fileTime)
        self.writerTime.writerow(ValueTime)

        fileTime.close()

    """
    写入数据
    """

    def writerVideo(self, FrameRate, W_Frame, H_Frame):
        videoName = self.videoName + '_' + str(self.userId) + 'Tk_EpoMax' + str(self.epochGCN) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.log_path + videoName, fourcc, FrameRate, (int(W_Frame), int(H_Frame)))
        return out

    def writerCSVT(self, content):
        fileTime = open(self.log_path + self.CSVfilenameTime, 'a', newline='')

        self.writerTime = csv.writer(fileTime)
        self.writerTime.writerow(content)

        fileTime.close()

    def writerCSVA(self, content):
        fileTime = open(self.log_path + self.CSVfilenameTime, 'a', newline='')

        self.writerTime = csv.writer(fileTime)
        self.writerTime.writerow(content)

        fileTime.close()

