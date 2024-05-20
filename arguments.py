from argparse import ArgumentParser

parser = ArgumentParser(description='Online viewport prediction with GNN')

parser.add_argument('--model', type=str, default='EvolveGCN-O',
                    help='We can choose EvolveGCN-O or EvolveGCN-H,'
                         'but the EvolveGCN-H performance on Elliptic dataset is not good.')
parser.add_argument('--n-hidden', type=int, default=128)
parser.add_argument('--n_output', type=int, default=200)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--cuda', default=True, type=bool, help='whether cuda is in use')
parser.add_argument('--policy', default='RGCN', type=str, help='select method')
parser.add_argument('--alpha', default=1, type=int, help='weight of negative sampling')
parser.add_argument('--beta', default=100, type=int, help='weight of stability')

# for 360 degree video
parser.add_argument('--sampleRate', default=10, type=int, help='how many frame of each')
parser.add_argument('--window', default=8, type=int, help='the size of window')
parser.add_argument('--videoId', default=1, type=int, help='select video')
parser.add_argument('--visId', default=-1, type=int, help='visualization user ID')
parser.add_argument('--epochGCN', default=20, type=int, help='the epoch for GCN')
parser.add_argument('--tileNum', default=5, type=int, help='number of tile for each row')

# cluster
parser.add_argument('--threshold', default=180, type=int, help='threshold of viewing similarity')
parser.add_argument('--thred', default=40, type=int, help='threshold of viewing similarity')

# for GCN train and test
parser.add_argument('--input_dim', default=200, type=int, help='')
parser.add_argument('--trainNum', default=43, type=int, help='number of user for training')
parser.add_argument('--testNum', default=5, type=int, help='number of user for test')

# general parameters
parser.add_argument('--output_dim', default=200, type=int, help='Output embed size')
parser.add_argument('--n_component', default=10, type=int, help='n_component')

# GAT parameters
parser.add_argument('--num_heads', default=8, type=int, help='Number of heads in each GAT layer')

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