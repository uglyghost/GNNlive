from gym.envs.registration import register

register(
    id='MyEnv-v1',  # 环境名,版本号v0必须有
    entry_point='env.grid_video_v1:MyEnv'  # 文件夹名.文件名:类名
    # 根据需要定义其他参数
)
