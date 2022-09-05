from msilib.schema import Class
from queue import Empty
from re import X
import matplotlib.pyplot as plt

class metrics():
    def __init__(self, episodes, rewards1, rewards2):
        self.episodes = episodes
        self.rewards1 = rewards1
        self.rewards2 = rewards2
        self.nonDominatedPoints = []
        self.ndPoints =[]
        self.pdict = {}

    def plotGraph(self):
        
        
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Rewards 1 x Episodes')
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Rewards 2 x Episodes')
        plt.show()
        
        

    def plot_pareto_frontier(self,poldict):
        '''Pareto frontier selection process'''
        self.pdict = poldict
        #print(self.pdict)
        i = 0

        
        #print(self.pdict[i])
        c = self.pdict[i]
       
        x = []
        y = []
        
        for v in self.pdict.values():
           for i in range(0,4):
            x.append(v[i][0][0])
            y.append(v[i][0][1])
        
       





        
        #ax.plot(self.rewards1, self.rewards2,'bx')
        fig, ax = plt.subplots()
        ax.plot(x,y,'ro')
        
        #plt.scatter(x,y)
        ax.set_title('Pareto Frontier')
        plt.xlabel("Treasure value")
        plt.ylabel("Time penalty")
        
        plt.show()
        
    def plot_pareto_frontier2(self,poldict, maxX=True, maxY=True):
        '''Pareto frontier selection process'''
        self.pdict = poldict
        #print(self.pdict)
        i = 0

        
        #print(self.pdict[i])
        c = self.pdict[i]
       
        x = []
        y = []
        
        for v in self.pdict.values():
           for i in range(0,4):
            #print(i)
            x.append(v[i][0][0])
            y.append(v[i][0][1])
        
        
            
        
        
        print(x)
        print(y)
        Xs = x
        Ys = y
        sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if maxY:
                if pair[1] >= pareto_front[-1][1]:
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)
        
        '''Plotting process'''
        plt.scatter(Xs,Ys)
        pf_X = [pair[0] for pair in pareto_front]
        pf_Y = [pair[1] for pair in pareto_front]
        plt.plot(pf_X, pf_Y)
        plt.xlabel("Treasure Reward")
        plt.ylabel("Time Penalty")
        plt.show()