from msilib.schema import Class
import matplotlib.pyplot as plt

class metrics():
    def __init__(self, episodes, rewards1, rewards2):
        self.episodes = episodes
        self.rewards1 = rewards1
        self.rewards2 = rewards2

    def plotGraph(self):
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Rewards 1 x Episodes')
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Rewards 2 x Episodes')
        plt.show()