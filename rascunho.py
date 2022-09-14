from multiprocessing.resource_sharer import stop
from pathlib import Path
from xml.etree.ElementTree import tostring

import gym
import numpy as np
from scipy import rand
#from sympy import Q
import pygame
from gym.spaces import Box, Discrete
from pygmo import hypervolume
from metrics import metrics as mtc

metrics = mtc([], [], [])

# As in Yang et al. (2019):
DEFAULT_MAP = np.array(
            [[0,    0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [0.7,  0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10,  8.2,  0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, 11.5,  0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, -10, 14.0, 15.1,16.1,0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3,0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10,  0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )

# As in Vamplew et al. (2018):
CONCAVE_MAP = np.array(
            [[0,    0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [1.0,  0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10,  2.0,  0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10,  3.0,  0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, -10, 5.0,  8.0,16.0, 0 ,  0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10, 24.0, 50.0,0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10,  0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 74.0, 0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 124.0,0]]
        )

class DeepSeaTreasure(gym.Env):
    """Deep Sea Treasure environment
    Adapted from: https://github.com/RunzheYang/MORL
    CCS weights: [1,0], [0.7,0.3], [0.67,0.33], [0.6,0.4], [0.56,0.44], [0.52,0.48], [0.5,0.5], [0.4,0.6], [0.3,0.7], [0, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, dst_map=DEFAULT_MAP, float_state=False):
        self.size = 11
        self.window_size = 512
        self.window = None
        self.clock = None
        self.epsilon = 0.99
        self.epsilonDecrease = 0.9
        self.paretoFront=[]
        self.paretoList = []
        
        self.paretoFrontResult = []

        self.float_state = float_state
        self.nA = 0

        self.stateList = []
        

        # The map of the deep sea treasure (convex version)
        self.sea_map = dst_map
        assert self.sea_map.shape == DEFAULT_MAP.shape, "The map shape must be 11x11"
        
        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32)  # right
        }

        # state space specification: 2-dimensional discrete box
        obs_type = np.float32 if self.float_state else np.int32
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=obs_type)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        self.reward_space = Box(low=np.array([0, -1]), high=np.array([23.7, -1]), dtype=np.float32)

        self.current_state = np.array([0, 0], dtype=np.int32)


    def get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]


    def is_valid_state(self, state):
        if state[0] >= 0 and state[0] <= 10 and state[1] >= 0 and state[1] <= 10:
            if self.get_map_value(state) != -10:
                return True
        return False
    

    def render(self, mode='human'):
        # The size of a single grid square in pixels
        pix_square_size = self.window_size / self.size
        if self.window is None:
            self.submarine_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/submarine.png')
            self.submarine_img = pygame.transform.scale(self.submarine_img, (pix_square_size, pix_square_size))
            self.submarine_img = pygame.transform.flip(self.submarine_img, flip_x=True, flip_y=False)
            self.treasure_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/treasure.png')
            self.treasure_img = pygame.transform.scale(self.treasure_img, (pix_square_size, pix_square_size))

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 30)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 105, 148))

        for i in range(self.sea_map.shape[0]):
            for j in range(self.sea_map.shape[1]):
                if self.sea_map[i,j] == -10:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([j,i]) + 0.6,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.sea_map[i,j] != 0:
                   canvas.blit(self.treasure_img, np.array([j,i]) * pix_square_size)
                   img = self.font.render(str(self.sea_map[i,j]), True, (255, 255, 255))
                   canvas.blit(img, np.array([j,i]) * pix_square_size + np.array([5, 20]))
 
        canvas.blit(self.submarine_img, self.current_state[::-1] * pix_square_size)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def get_state(self):
        s = ''.join(str(self.current_state))

        if s not in self.stateList:
            self.stateList.append(s)
        
        return self.stateList.index(s)


    def reset(self, seed=None, return_info=False, **kwargs):
        super().reset(seed=seed)
        self.np_random.seed(seed)

        self.current_state = np.array([0, 0], dtype=np.int32)
        #self.current_state = 0
        self.step_count = 0.0
        state = self.get_state()

        return (state, {}) if return_info else state


    def step(self, action):
        next_state = self.current_state + self.dir[action]

        if self.is_valid_state(next_state):
            self.current_state = next_state

        treasure_value = self.get_map_value(self.current_state)
        if treasure_value == 0 or treasure_value == -10:
            treasure_value = 0.0
            terminal = False
        else:
            terminal = True
        time_penalty = -1.0
        vec_reward = np.array([treasure_value, time_penalty], dtype=np.float32)

        state = self.get_state()

        return state, vec_reward, terminal, {}


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class Pareto(DeepSeaTreasure):
    def __init__(self, env, choose_action, ref_point, nO=2, gamma=1.):
        self.env = env
        self.choose_action = choose_action
        self.gamma = gamma

        self.ref_point = ref_point

        self.nS = 64
        
        self.nA = env.action_space.n
        env.nA = self.nA
        self.non_dominated = [[[np.zeros(nO)] for _ in range(self.nA)] for _ in range(self.nS)]
        self.avg_r = np.zeros((self.nS, self.nA, nO))
        self.n_visits = np.zeros((self.nS, self.nA))
        self.epsilon = 0
        

    def initializeState(self):
        state = self.env.reset()
        
        return {'observation':state,'terminal':False}


    def train(self,max_episodes,max_steps):
        numberOfEpisodes = 0
        episodeSteps = 0

        #line 1 -> initialize q_set
        print("-> Training started <-")
        #line 2 -> for each episode
        while numberOfEpisodes  < max_episodes:

            acumulatedRewards = [0,0]
            episodeSteps = 0

            #line 3 -> initialize state s
            s = self.initializeState()
            
            #line 4 and 11 -> repeat until s is terminal:
            while s['terminal'] is not True and episodeSteps < max_steps:
                #env.render()
                s = self.step(s)
                print(s, episodeSteps)
                episodeSteps += 1
                acumulatedRewards[0] += s['reward'][0]
                acumulatedRewards[1] += s['reward'][1]

            metrics.rewards1.append(acumulatedRewards[0])
            metrics.rewards2.append(acumulatedRewards[1])
            metrics.episodes.append(numberOfEpisodes)
            numberOfEpisodes+=1
            print(numberOfEpisodes)
            metrics.plot_pareto_frontier2(self.polDict)


    def step(self,state):
        s = state['observation']

        #line 5 -> Choose action a from s using a policy derived from the Qˆset’s
        
        q_set = self.compute_q_set(s)
        action = self.choose_action(s, q_set)
        self.qcopy = copy.deepcopy(q_set)
        self.polDict[self.polIndex] = self.qcopy
        self.polIndex +=1
        #line 6 ->Take action a and observe state s0 ∈ S and reward vector r ∈ R
        next_state, reward, terminal, _ = self.env.step(action)
        
        #line 8 -> . Update ND policies of s' in s
        self.update_non_dominated(s, action, next_state)
        
        #line 9 -> Update avg immediate reward
        self.n_visits[s, action] += 1

        self.avg_r[s, action] += (reward - self.avg_r[s, action]) / self.n_visits[s, action]

        env.epsilon *= 0.999

        return {'observation': next_state,
                'terminal': terminal,
                'reward': reward}

    
    def compute_q_set(self, s):
        q_set = []
        for a in range(self.env.nA):
            nd_sa = self.non_dominated[s][a]
            rew = self.avg_r[s, a]
            q_set.append([rew + self.gamma*nd for nd in nd_sa])
        return np.array(q_set)


    def update_non_dominated(self, s, a, s_n):
        q_set_n = self.compute_q_set(s_n)
        # update for all actions, flatten
        solutions = np.concatenate(q_set_n, axis=0)
        # append to current pareto front
        # solutions = np.concatenate([solutions, self.non_dominated[s][a]])

        # compute pareto front
        self.non_dominated[s][a] = get_non_dominated(solutions)

def get_action(s, q,env):
    q_values = compute_hypervolume(q, q.shape[0], ref_point)

    if np.random.rand() >= env.epsilon:
        return np.random.choice(np.argwhere(q_values == np.amax(q_values)).flatten())
    else:
        return env.action_space.sample()

def compute_hypervolume(q_set, nA, ref):
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        points = np.array(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values

def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1

    return solutions[is_efficient]

if __name__ == '__main__':
    import gym
    from gym import wrappers

    env = DeepSeaTreasure()
    ref_point = np.array([0, -25])
    agent = Pareto(env, lambda s, q: get_action(s, q, env), ref_point, nO=2, gamma=1.)
    #print("oi aqui")
    agent.train(1000,200)
    metrics.plotGraph()