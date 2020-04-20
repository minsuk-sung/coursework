'''
USAGE: python3 run.py --example AND
'''

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Deep Learning HW1')
	parser.add_argument('--example', required=True,help='AND, OR 선택가능')
	args = parser.parse_args()

	example = args.example
	y = {'AND':np.array([-1,-1,-1,1]),'OR':np.array([-1,1,1,1]),'XOR':np.array([-1,1,1,-1])}

	X = np.array([[0,0],[0,1], [1,0], [1,1]])
	y = y[example]

	model = Perceptron(eta=0.01,n_iter=100,example=example)
	print('Model Training...')
	model.fit(X,y)
	print('Weights : ',model.w_)

	print('Saving Decision Boundary...')
	model.save_gif_decision_boundary()
	print('Complete saving GIF file')