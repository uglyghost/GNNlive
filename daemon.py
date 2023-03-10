import os
from arguments import get_args

args = get_args()

for num in range(1,9):
    command = 'python main.py ' + '--videoId ' + str(num)
    os.system(command)