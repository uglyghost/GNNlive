from argparse import ArgumentParser

parser = ArgumentParser(description='Online viewport prediction with GNN')


parser.add_argument('--cuda', default='GPU', type=bool, help='whether cuda is in use')

# for 360 degree video
parser.add_argument('--sampleRate', default=1, type=int, help='how many frame of each')
parser.add_argument('--videoId', default=1, type=int, help='select video')
parser.add_argument('--visId', default=8, type=int, help='visualization user ID')
parser.add_argument('--epochGCN', default=10, type=float, help='the epoch for GCN')

# cluster
parser.add_argument('--threshold', default=120, type=int, help='threshold of viewing similarity')

# for GCN train and test
parser.add_argument('--trainNum', default=5, type=int, help='number of user for training')
parser.add_argument('--testNum', default=10, type=int, help='number of user for test')

# dataset path settings
parser.add_argument('--records_path', default='D:/Multimedia/FoV_Prediction/Dataset/VRdataset/Experiment_',
                    type=str, metavar='PATH',
                    help='path to users viewport records')
parser.add_argument('--videos_path', default='D:/Multimedia/FoV_Prediction/Dataset/Videos/',
                    type=str, metavar='PATH',
                    help='path to video')
parser.add_argument('--frames_path', default='D:/Multimedia/FoV_Prediction/Dataset/frames/',
                    type=str, metavar='PATH',
                    help='path to video')

# other path settings
parser.add_argument('--model_path', default='./save_model/',
                    type=str, metavar='PATH',
                    help='path to trained model')
parser.add_argument('--checkpoint_file', default='./save_model/checkpoint.pth.tar',
                    type=str, metavar='PATH',
                    help='')
parser.add_argument('--log_path', default='./log/',
                    type=str, metavar='PATH',
                    help='path to save log csv')

args = parser.parse_args()


def get_args():
    arguments = parser.parse_args()
    return arguments