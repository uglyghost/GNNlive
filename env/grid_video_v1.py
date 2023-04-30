import logging
import random
import gym
import numpy as np
import math

logger = logging.getLogger(__name__)


class MyEnv(gym.Env):

    metadata = {
        'render.modes': ['liveVideo', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self):

        self.step = 0
        self.states = np.zeros([400])               # 8帧历史视野和8帧显著性特征 共 (8+8)*25 = 400
        self.actions = np.arange(0, 24, 1)          # 设置视野所处的tile的索引为动作 共25tile 25个动作

        self.gamma = 0.8                            # 折扣因子
        self.weightBand = 0.2
        self.viewer = None
        self.state = None
        self.viewpoint = []
        self.prediction = []

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setState(self, s):
        self.state = s

    def setPrediction(self, p):
        self.prediction = p

    def getRewardT(self, action, next_state):
        TP, TN, FP, FN = 0, 0, 0, 0
        for i, value in enumerate(action):
            if value + next_state[i] == 2:
                TP += 1
            elif value + next_state[i] == 0:
                TN += 1
            elif value == 0 and next_state[i] == 1:
                FP += 1
            else:
                FN += 1

        bandwidth = FP + FN
        experience = TP - FP
        reward = experience - self.weightBand*bandwidth

        return reward

    def getNextSate(self, solver):
        next_state = []
        if solver.countMain + solver.bufLen < solver.totalFrames:
            for i in range(solver.bufLen):
                if i % solver.sampleRate == 0:
                    next_state.append(solver.LocationPerFrame[math.ceil(solver.interFOV * (solver.countMain - 1)) + 1])
                solver.countMain += 1
            return next_state
        else:
            return next_state

    def reset(self):
        self.state = None
        return self.state

    def setFov(self, viewpoint):
        self.viewpoint = viewpoint
        return self.viewpoint

    def render(self, mode='liveVideo'):
        from gym.envs.classic_control import rendering
        screen_width = 1400
        screen_height = 700
        screen_width_ac = screen_width - 200
        screen_height_ac = screen_height - 100

        table = [[[(200, 100), (400, 100), (400, 200), (200, 200)], [(400, 100), (600, 100), (600, 200), (400, 200)],
                  [(600, 100), (800, 100), (800, 200), (600, 200)], [(800, 100), (1000, 100), (1000, 200), (800, 200)],
                  [(1000, 100), (1200, 100), (1200, 200), (1000, 200)]],
                 [[(200, 200), (400, 200), (400, 300), (200, 300)], [(400, 200), (600, 200), (600, 300), (400, 300)],
                  [(600, 200), (800, 200), (800, 300), (600, 300)], [(800, 200), (1000, 200), (1000, 300), (800, 300)],
                  [(1000, 200), (1200, 200), (1200, 300), (1000, 300)]],
                 [[(200, 300), (400, 300), (400, 400), (200, 400)], [(400, 300), (600, 300), (600, 400), (400, 400)],
                  [(600, 300), (800, 300), (800, 400), (600, 400)], [(800, 300), (1000, 300), (1000, 400), (800, 400)],
                  [(1000, 300), (1200, 300), (1200, 400), (1000, 400)]],
                 [[(200, 400), (400, 400), (400, 500), (200, 500)], [(400, 400), (600, 400), (600, 500), (400, 500)],
                  [(600, 400), (800, 400), (800, 500), (600, 500)], [(800, 400), (1000, 400), (1000, 500), (800, 500)],
                  [(1000, 400), (1200, 400), (1200, 500), (1000, 500)]],
                 [[(200, 500), (400, 500), (400, 600), (200, 600)], [(400, 500), (600, 500), (600, 600), (400, 600)],
                  [(600, 500), (800, 500), (800, 600), (600, 600)], [(800, 500), (1000, 500), (1000, 600), (800, 600)],
                  [(1000, 500), (1200, 500), (1200, 600), (1000, 600)]]]

        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)

            #创建网格世界
            self.line1 = rendering.Line((200, 100), (1200, 100))
            self.line2 = rendering.Line((200, 200), (1200, 200))
            self.line3 = rendering.Line((200, 300), (1200, 300))
            self.line4 = rendering.Line((200, 400), (1200, 400))
            self.line5 = rendering.Line((200, 500), (1200, 500))
            self.line6 = rendering.Line((200, 600), (1200, 600))
            self.line7 = rendering.Line((200, 100), (200, 600))
            self.line8 = rendering.Line((400, 100), (400, 600))
            self.line9 = rendering.Line((600, 100), (600, 600))
            self.line10 = rendering.Line((800, 100), (800, 600))
            self.line11 = rendering.Line((1000, 100), (1000, 600))
            self.line12 = rendering.Line((1200, 100), (1200, 600))

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)

        else:
            # self.viewer = rendering.Viewer(screen_width, screen_height)

            if self.state is None:
                return None

            # self.viewpoint
            '''
            point_array = []
            '''
            for index, value in enumerate(self.viewpoint):
                oneFrame = []
                point_array = []
                point_array.append((value[0] * screen_width_ac - 120, value[1] * screen_height_ac + 60))
                point_array.append((value[0] * screen_width_ac + 120, value[1] * screen_height_ac + 60))
                point_array.append((value[0] * screen_width_ac + 120, value[1] * screen_height_ac - 60))
                point_array.append((value[0] * screen_width_ac - 120, value[1] * screen_height_ac - 60))

                viewport = rendering.make_polygon(point_array)
                circletrans = rendering.Transform(translation=(75, 150))
                viewport.set_color(0, 0, 1)
                viewport.add_attr(circletrans)

                for i in range(5):
                    for j in range(5):
                        if int(self.prediction[(index - 1) * 25 + (i - 1) * 5 + j]) == 1:
                            prediction = rendering.make_polygon(table[i][j])
                            prediction.set_color(1, 0, 0)
                            oneFrame.append(prediction)
                oneFrame.append(viewport)

                for k, value in enumerate(oneFrame):
                    self.viewer.add_onetime(value)
                # self.viewer.add_onetime(self.prediction)
                # self.viewer.add_onetime()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()