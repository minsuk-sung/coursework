import os
import glob
import numpy as np
import matplotlib.pyplot
import imageio
from tqdm import tqdm
fig =  matplotlib.pyplot.figure()
plt = fig.add_subplot(111)

# 퍼셉트론 클래스
class Perceptron():
    # 초기화
    def __init__(self,example,thresholds=0.0,eta=0.01,n_iter=10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
        self.example = example
        self.gif_path = './img/{}/'.format(example)

    # 학습
    def fit(self,X,y):
        self.w_ = np.random.normal(size=1+X.shape[1])
        self.errors_ = []

        for _iter in tqdm(range(self.n_iter)):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (self.predict(xi)-target)
                self.w_[1:] -= update * xi
                self.w_[0] -= update
                errors += int(update!=0.0)
            self.errors_.append(errors)
            self.show_decision_boundary(X,y,_iter)
        return self

    # 추론
    def predict(self,X):
        return np.sum(X*self.w_[1:])+self.w_[0]

    # 
    def show_weight(self):
        print(self.w_)

    def show_decision_boundary(self,X,y,_iter):
        markers = {1:'o',-1:'x'}
        colors = ['b','r']
        for xs,ys in zip(X,y):
            if self.predict(xs) > self.thresholds:
                plt.plot(xs[0],xs[1],markers[ys],c=colors[0])
            else:
                plt.plot(xs[0],xs[1],markers[ys],c=colors[1])
        matplotlib.pyplot.xlim([-0.25,1.25])
        matplotlib.pyplot.ylim([-0.25,1.25])
        plt.grid(True)
        _x = np.arange(-10,10)
        _y = -(self.w_[0] / self.w_[2]) - (self.w_[1]/self.w_[2])*_x
        plt.plot(_x, _y)
        matplotlib.pyplot.title('[{}] Decision Boundary: {} Gate'.format(_iter,self.example))
        matplotlib.pyplot.savefig(self.gif_path+'{}.jpg'.format(str(_iter).zfill(4)))
        matplotlib.pyplot.pause(0.000000001)
        plt.lines.pop(-1)

    def save_gif_decision_boundary(self):
        images = []
        for filename in tqdm(sorted(glob.glob(self.gif_path+'/*.jpg'))):
            images.append(matplotlib.pyplot.imread(filename))
        imageio.mimsave('./{}.gif'.format(self.example), images)
