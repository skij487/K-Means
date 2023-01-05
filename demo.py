import numpy as np
import matplotlib.pyplot as plt
import KMeans

points = np.loadtxt('POIs')
K = 3
zs, cost, mus = KMeans.KMeansMin(points, K)
colors = [(1,0,0), (0,1,0), (0,0,1)]
for i in range(K):
        plt.plot(points[zs == i,0],points[zs == i,1],'.',color=colors[i],markersize=8)
        plt.plot(mus[i][0],mus[i][1],'k*',markersize=16)
plt.show()