import matplotlib.pyplot as plt
import numpy as np

tmp_x = np.linspace(-1,1,50)
tmp_y = 2*tmp_x +1
amp_x = np.linspace(-1,2,50)
amp_y = 3*amp_x +1

def plot_points(tmpx, tmpy,tfx,tfy):
    plt.figure(figsize=(11,11))
    # 子表格为1*1，选择第一个。 坐标轴范围x:[-50,250];y:[50,350]
    ax = plt.subplot(111)
    ax.axis([-1, 5, -2, 10])
    ax.scatter(tmpx, tmpy, c='red')
    ax.scatter(tfx,tfy,c='green')
    for index in range(len(tmpx)):
        # 添加一个text备注，坐标为(tmpx,tmpy)，内容为index
        ax.text(tmpx[index], tmpy[index], index)
    for index in range(len(tfx)):
        ax.text(tfx[index],tfy[index],index)
    plt.show()

plot_points(tmp_x,tmp_y,amp_x,amp_y)

vehicle = 'vehicle.tesla.model3_122'
print(vehicle[-3:])
str = '123.ply'
print(str[:-3])

