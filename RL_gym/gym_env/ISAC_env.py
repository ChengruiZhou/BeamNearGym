import gym
from gym import error, spaces, utils
from gym.utils import seeding
import globe
import numpy as np
import matplotlib.pyplot as plt
import math



''' 
    Environment parameters
        cell radius
        UE movement speed
        BS max tx power
        BS antenna
        UE noise figure
        Center frequency
        Transmit antenna isotropic gain
        Antenna heights
        Shadow fading margin
        Number of ULA antenna elements
        Oversampling factor
'''


class ISACEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''    
            Observation: 
                Type: Box(6 or 8)
                Num Observation                                    Min      Max
                0   User1 server X                                 -r       r
                1   User1 server Y                                 -r       r
                2   User2 server X                                 -r       r
                3   User2 server Y                                 -r       r
                4   Serving BS Power                               5        40W
                5   Neighbor BS power                              5        40W
                6   BF codebook index for Serving                  0        M-1
                7   BF codebook index for Neighbor                 0        M-1
    '''
    def __init__(self):
        super(ISACEnv, self).__init__()
        globe._init()

        globe.set_value('dx', 0.5)  # x element spacing
        globe.set_value('dy', 0.5)  # y element spacing
        globe.set_value('theta0', 45 * (np.pi / 180))  # Elevation angle
        globe.set_value('phi0', 30 * (np.pi / 180))  # Azimuth angle
        globe.set_value('range_val', 10)  # Range distance0        globe.set_value('lamda', 0.125)

        globe.set_value('cell_radius', 150) # in meters.
        globe.set_value('inter_site_distance', 3 * globe.get_value('cell_radius') / 2.)

        globe.set_value('M', 20)  # Number of elements in x
        globe.set_value('N', 20)  # Number of elements in y

        globe.set_value('v', 8)  # Move speed / velocity (m/s)

        globe.set_value('num_users', 2)  # number of users.

        self.alpha_x = 2 * np.pi * np.sin(np.deg2rad(globe.get_value('phi0'))) * np.cos(np.deg2rad(globe.get_value('theta0')))  # x phase difference
        self.alpha_y = 2 * np.pi * np.sin(np.deg2rad(globe.get_value('theta0')))  # y phase difference

        # Element positions M*N dim
        self.X = np.arange(0, globe.get_value('M')) * globe.get_value('dx')  # x array
        self.Y = np.arange(0, globe.get_value('N')) * globe.get_value('dy')  # y array
        self.X2 = np.kron(np.ones(globe.get_value('N')), self.X)  # Repeat x array for each y
        self.Y2 = np.kron(self.Y, np.ones(globe.get_value('M')))  # Repeat y array for each x

        # Steering Vectors
        self.ax = np.exp((complex(0, 1)) * self.X * self.alpha_x)
        self.ay = np.exp((complex(0, 1)) * self.Y * self.alpha_y)
        self.axy = np.kron(self.ax, self.ay)

        # the location of the base station
        self.x_bs, self.y_bs = 0, 0

        self.state = None
        self.np_random = None

        self.positions = {"UE1": [], "UE2": []}
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(2)  # Example: 2 possible actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Example: 3-dimensional observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.random.rand(3)
        # return self.state
        self.state = [self.np_random.uniform(low=-globe.get_value('cell_radius'), high=globe.get_value('cell_radius')),
                      self.np_random.uniform(low=-globe.get_value('cell_radius'), high=globe.get_value('cell_radius')),
                      self.np_random.uniform(low=-globe.get_value('cell_radius'), high=globe.get_value('cell_radius')),
                      self.np_random.uniform(low=-globe.get_value('cell_radius'), high=globe.get_value('cell_radius'))]
        return np.array(self.state)


    def step(self, action):
        # Execute one time step within the environment
        # reward = 1 if action == 1 else 0  # Example: reward logic
        # done = np.random.rand() > 0.95  # Example: end condition
        # self.state = np.random.rand(3)  # Example: new state
        # return self.state, reward, done, {}

        state = self.state
        reward = 0
        x_ue_1, y_ue_1, x_ue_2, y_ue_2 = state
        # move the UEs at a speed of v, in a random direction
        theta_1, theta_2 = self.np_random.uniform(low=-math.pi, high=math.pi, size=2)
        dx_1 = globe.get_value('v') * math.cos(theta_1)
        dy_1 = globe.get_value('v') * math.sin(theta_1)

        dx_2 = globe.get_value('v') * math.cos(theta_2)
        dy_2 = globe.get_value('v') * math.sin(theta_2)

        # Move UE 1
        x_ue_1 += dx_1
        y_ue_1 += dy_1

        # Move UE 2
        x_ue_2 += dx_2
        y_ue_2 += dy_2

        # 记录位置
        self.positions["UE1"].append((x_ue_1, y_ue_1))
        self.positions["UE2"].append((x_ue_2, y_ue_2))

        # 更新状态
        self.state = [x_ue_1, y_ue_1, x_ue_2, y_ue_2]

        # 示例返回
        reward = 0
        done = False
        return np.array(self.state), reward, done, {}

    def plot_positions(self):
        # 提取 UE1 和 UE2 的轨迹
        ue1_x, ue1_y = zip(*self.positions["UE1"])
        ue2_x, ue2_y = zip(*self.positions["UE2"])

        plt.figure(figsize=(10, 6))
        plt.plot(ue1_x, ue1_y, '-o', label="UE1 Trajectory")
        plt.plot(ue2_x, ue2_y, '-o', label="UE2 Trajectory")

        # 标注时间步
        for i, (x, y) in enumerate(self.positions["UE1"]):
            plt.text(x, y, str(i), fontsize=8, color="blue")
        for i, (x, y) in enumerate(self.positions["UE2"]):
            plt.text(x, y, str(i), fontsize=8, color="orange")

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Trajectories of UE1 and UE2")
        plt.legend()
        plt.grid()
        plt.show()
        # return np.array(self.state), reward, done, abort


    def render(self, mode='human'):
        # Render the environment to the screen
        print(f'State: {self.state}')

    def close(self):
        pass

    def plotSystem(self):
        # Plotting
        plt.figure()
        plt.plot(self.X2, self.Y2, '.')
        plt.axis('equal')
        plt.grid(True)
        plt.title('Antenna Array')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        print('Fig the system UPA figure')

    def cal_channel_UPW(self):
        print('cal_channel_UPW')

    def cal_channel_USW(self):
        print('cal_channel_USW')

    def cal_channel_NUSW(self):
        print('cal_channel_NUSW')

    def cal_channel_ACC(self):
        print('cal_channel_ACC')
