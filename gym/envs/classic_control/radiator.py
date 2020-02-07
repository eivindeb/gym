import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import matplotlib.pyplot as plt

class RadiatorEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.dt=.05
        self.heat_capacity = 1.5
        self.heat_coeff = 0.2
        self.radiator_area = 10
        self.time_constant = self.heat_capacity / (self.radiator_area * self.heat_coeff)
        self.viewer = None
        self.last_u = None

        self.hist = []

        self.min_room_temp = 0
        self.max_room_temp = 40

        self.min_radiator_temp = 0
        self.max_radiator_temp = 30

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([self.min_room_temp, self.min_radiator_temp]),
                                            high=np.array([self.max_room_temp, self.max_radiator_temp]),
                                            dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = np.clip(u, self.action_space.low, self.action_space.high)
        u = (self.max_radiator_temp - self.min_radiator_temp) / 2 * (u + 1) + self.min_radiator_temp
        if self.last_u is None:
            self.last_u = u
        cur_temp, desired_temp = self.state

        cur_temp += (- 1 / self.time_constant * (u - desired_temp)) * self.dt

        cur_temp = np.clip(cur_temp, self.min_room_temp, self.max_room_temp)

        costs = 1 / (self.max_room_temp - self.min_room_temp) ** 2 * (cur_temp - desired_temp) ** 2 + np.abs((u - self.last_u)) * 0.01

        self.last_u = u

        self.state = np.array([cur_temp[0], desired_temp])
        self.hist.append(self.state)

        return self.state, -costs[0], False, {}

    def reset(self, state=None):
        if state is None:
            self.state = self.np_random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        else:
            self.state = state
        self.hist = []
        self.last_u = None
        return self.state

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human', title=None):
        data = np.array(self.hist)
        x = np.arange(data.shape[0])
        plt.plot(x, data[:, 0], label="Current")
        plt.plot(x, data[:, 1], label="Desired")
        plt.title(title)
        plt.legend()
        plt.show()

        return
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)