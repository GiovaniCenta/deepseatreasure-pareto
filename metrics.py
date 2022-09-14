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
        self.xA0 = []
        self.yA0 = []        
        self.xA1 = []
        self.yA1 = []
        self.xA2 = []
        self.yA2 = []
        self.xA3 = []
        self.yA3 = []

    def plotGraph(self):
        
        
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Rewards 1 x Episodes')
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Rewards 2 x Episodes')
        plt.show()
    
    def plot_p_front(self,Xs,Ys,actionIndex,maxY = True,maxX = True):
        
         
        sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if maxY:
                best = []
                if pair[1] >= pareto_front[-1][1]:
                    #print(pair[1])
                    """
                    backup = pair[1]
                    best.append(backup)
                    exist_count = pareto_front.count(backup)
                    print(exist_count)
                    
                    if exist_count > 0:
                        print("entrou aqui")
                        """

                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)
        
        '''Plotting process'''
        plt.scatter(Xs,Ys)
        pf_X = [pair[0] for pair in pareto_front]
        pf_Y = [pair[1] for pair in pareto_front]
        plt.plot(pf_X, pf_Y)
        plt.xlabel("Treasure Reward for Action " + str(actionIndex) )
        plt.ylabel("Time Penalty for Action " + str(actionIndex))
        plt.show()     
        
        

    def plot_p_front2(self,x,y,indexAction):
        Xs, Ys = x,y
    # Find lowest values for cost and highest for savings
        p_front = self.pareto_frontier(Xs, Ys, maxX = True, maxY = True) 
        # Plot a scatter graph of all results
        plt.scatter(Xs, Ys)
        # Then plot the Pareto frontier on top
        plt.plot(p_front[0], p_front[1])
        plt.show()
        """
        fig, ax = plt.subplots()
        ax.plot(x,y,'ro')
        
        #plt.scatter(x,y)
        ax.set_title('Pareto Frontier')
        plt.xlabel("Treasure value")
        plt.ylabel("Time penalty")
        
        plt.show()
        """

    def pareto_frontier(self,Xs, Ys, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]    
    # Loop through the sorted list
        for pair in myList[1:]:
            best =[]
            if maxY: 
                if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                    best.append(pair[1])
                    if pair[1] in best:
                        print(max(best))
                        p_front.append(pair)
                     # … and add them to the Pareto frontier
                    
                    
            else:
                if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                    p_front.append(pair) # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
        #exit(8)
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return p_frontX, p_frontY


        
    def plot_pareto_frontier2(self,poldict, maxX=True, maxY=True):
        '''Pareto frontier selection process'''
        self.pdict = poldict
        #print(self.pdict)
        i = 0

        
        #print(self.pdict[i])
        c = self.pdict[i]

        
        
        for v in self.pdict.values():
            self.xA0.append(v[0][0][0])
            self.yA0.append(v[0][0][1])
        for v in self.pdict.values():
            self.xA1.append(v[1][0][0])
            self.yA1.append(v[1][0][1])
        for v in self.pdict.values():
            self.xA2.append(v[2][0][0])
            self.yA2.append(v[2][0][1])
        for v in self.pdict.values():
            self.xA3.append(v[3][0][0])
            self.yA3.append(v[3][0][1])
        
        #print(xA0)
        self.plot_p_front2(self.xA0,self.yA0,0)
        self.plot_p_front2(self.xA1,self.yA1,1)
        self.plot_p_front2(self.xA2,self.yA2,2)
        self.plot_p_front2(self.xA3,self.yA3,3)
        
        
    