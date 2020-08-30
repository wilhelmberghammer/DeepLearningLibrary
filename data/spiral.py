import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def twospirals(n_points, noise = 1):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))

X, y_ = twospirals(2000)

#plt.plot(X[y_==0,0], X[y_==0,1], '.', color='green')
#plt.plot(X[y_==1,0], X[y_==1,1], '.', color='red')
#plt.show()

y =[] 
for i in range(len(y_)):
    y.append(int(y_[i]))
y = np.eye(2)[y]
