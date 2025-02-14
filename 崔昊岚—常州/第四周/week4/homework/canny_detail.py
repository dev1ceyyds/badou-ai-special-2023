import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

pic_path = 'lenna.png'
img = plt.imread(pic_path)
if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    img = img * 255  # 还是浮点数类型
img = img.mean(axis=-1)

dim=int(np.round(6*0.5+1)) #高斯滤波的的维度
if dim%2==0:
    dim+=1
Gaussian=np.zeros([dim,dim])

tmp=[i-dim//2 for i in range(dim)]
n1 = 1 / (2 * math.pi * 0.5 ** 2)  # 计算高斯核
n2 = -1 / (2 * 0.5 ** 2)
for i in range(dim):
    for j in range(dim):
        Gaussian[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
Gaussian = Gaussian/ Gaussian.sum()
img_new=np.zeros(img.shape)
tmp=dim//2
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian)

plt.figure(1)#创建一个新窗口编号为1
plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')#关掉x和y轴


#求梯度
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
img_tidu_y = np.zeros([img.shape[0], img.shape[1]])
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 因为是3*3所以补1
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_tidu_x[i,j]=np.sum(img_pad[i:i + 3, j:j+3]*sobel_kernel_x)
        img_tidu_y[i,j]=np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
        img_tidu[i,j]=np.sqrt(img_tidu_x[i,j]**2+img_tidu_y[i,j]**2)
img_tidu_x[img_tidu_x==0] = 0.0000001
angle=img_tidu_y/img_tidu_x
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')#关掉x和y轴

# 非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
dx=img.shape[0]
dy=img.shape[1]
for i in range(1, dx-1):
    for j in range(1, dy-1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

'''
  先判断这个点如果是高于高阈值，那就是第一批成功的点
    小于低阈值，那就滚吧
    接着按照这些高阈值的点四周的点，看看有没有在高低阈值之间的，有的话，就加入zhan，zhan里面每个点都会要判断周围点
    最后除了255的成功的，其他点都滚
'''
lower_boundary = img_tidu.mean() * 0.5
high_boundary = lower_boundary * 3
zhan = []
for i in range(1, img_yizhi.shape[0]-1):  # 外圈不考虑了
    for j in range(1, img_yizhi.shape[1]-1):
        if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:  # 舍
            img_yizhi[i, j] = 0
while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()  # 出栈
    a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
        zhan.append([temp_1-1, temp_2-1])  # 进栈
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

# 绘图
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  # 关闭坐标刻度值

plt.show()

