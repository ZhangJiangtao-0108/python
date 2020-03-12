import matplotlib.pyplot as plt 
from math import *
from numpy import *
import numpy as np
import matplotlib.image as mpimg

'''
# 柱状图
fig = plt.figure(figsize=(4,2))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
x = [0,1,2,3,4,5,6,7,8,9,10]
y1 = [0,1,2,3,4,5,6,7,8,9,10]
y2 = [0,1,2,3,4,5,6,7,8,9,10]
y3 = [0,1,2,3,4,5,6,7,8,9,10]
y4 = [0,1,2,3,4,5,6,7,8,9,10]
ax1.bar(x,y1)
ax1.set_title("figure1")
ax2.bar(x,y2)
ax2.set_title("figure2")
ax3.bar(x,y3)
ax3.set_title("figure3")
ax4.bar(x,y4)
ax4.set_title("figure4")
plt.show()
'''

'''
# 扇形图
y = [2, 3, 8.8, 6.6, 7.0]
plt.figure()
plt.pie(y)
plt.title('PIE')
plt.show()
'''
'''
# 散点图
x = [0,1,2,3,4,5,6,7,8,9,10]
y = [0,1,2,3,4,5,6,7,8,9,10]
plt.scatter(x, y, color='r', marker='+')
plt.show()
'''
'''
# 函数图
x = arange(-math.pi, math.pi, 0.01)
y = [sin(xx) for xx in x]
plt.figure()
plt.title("sinx")
plt.plot(x, y, color='r', linestyle='-.')
plt.show()
'''
'''
# 2D data
delta = 0.025
x = y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z = Y**1 + X**2
plt.figure(figsize=(12, 6))
plt.contour(X, Y, Z)
plt.colorbar()
plt.title("2D")
plt.show()
'''
'''
# 读取照片
img=mpimg.imread('图片路径')
plt.imshow(img)
plt.title("图片")
plt.show()
'''
