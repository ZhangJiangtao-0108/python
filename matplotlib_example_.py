import matplotlib.pyplot as plt 

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(4,2))
x = [0,1,2,3,4,5,6,7,8,9,10]
y1 = [0,1,2,3,4,5,6,7,8,9,10]
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(x,y1)
plt.xlabel("x坐标轴",fontsize=20)
plt.ylabel("y坐标轴",fontsize=20)
plt.annotate('注释', xy=(2, 1), xytext=(3, 4),color='r',size=15,
            arrowprops=dict(facecolor='g', shrink=0.05))
plt.title("figure")
plt.text(6, 5, "test", size=50, rotation=30.,ha="center", va="center",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
plt.show()