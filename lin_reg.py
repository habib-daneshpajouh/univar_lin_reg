#
# Created by: Habib Daneshpajouh
# habib.dpajouh@gmail.com
#

import numpy as np
#import matplotlib.pyplot as plt


def plot_data(x, y, title):

	plt.plot(x, y)
	plt.title(title)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

	return


# read data
skiprows = 1

x_tr = np.loadtxt(fname="x_train.csv", skiprows=skiprows)
y_tr = np.loadtxt(fname="y_train.csv", skiprows=skiprows)
x_tst = np.loadtxt(fname="x_test.csv", skiprows=skiprows)
y_tst = np.loadtxt(fname="y_test.csv", skiprows=skiprows)

m_tr = np.size(x_tr,0)
m_tst = np.size(x_tst,0)

print("# training examples: " + str(m_tr))
print("# test examples: " + str(m_tst) + "\n")

# plot data
#plot_data(x_tr, y_tr, "Training examples")
#plot_data(x_tst, y_tst, "Test examples")

# init/define parameters
alpha = 0.001
theta0 = 0.0
theta1 = 0.0

num_itr = 50

# training loop
for i in range (num_itr):
	hx_tr = theta0 + theta1 * x_tr
	cost_tr = sum(pow(hx_tr - y_tr, 2)) / (2*m_tr)
	
	# gradient descent update
	theta0 = theta0 - (sum((hx_tr - y_tr)) / m_tr) * alpha
	theta1 = theta1 - (sum(((hx_tr - y_tr) * x_tr)) / m_tr) * alpha

	# print stats
	if((i+1)%10 == 0):
		print("Error after " + str(i+1) + " iterations: " + str(cost_tr))

	if(i == num_itr-1):
		print("\nTraining error: " + str(cost_tr))

# try with test data
hx_tst = theta0 + theta1 * x_tst
cost_tst = sum(pow(hx_tst - y_tst, 2)) / (2*m_tst)

print("Test error: " + str(cost_tst))
