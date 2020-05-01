import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# 第一步 数据选取
d = datasets.load_iris() # 使用鸢尾花数据集
x = d.data              # 样本自变量
y = d.target            # 样本标签
x = x[y<2,:2]           # 只取前两类
y = y[y<2]

# 第二步 数据预处理，标准化
s1 = StandardScaler()
s1.fit(x) # 将原数据进行标准化
x_standard = s1.transform(x)

# 第三步 定义模型
s11 = LinearSVC(C = 1e9)

# 第四步 拟合数据
s11.fit(x_standard, y)

# 第五步 可视化决策边界
plt.figure(figsize = (12, 12))
def plot_svc_decision_boundary(model, axis):
    '''
    Plot the classification result and support vectors
    Input:
        model: the trained svm model
        axis: the x, y range of the graph, should be 
    '''
    x0, x1 = np.meshgrid( # 绘图范围的坐标矩阵，100*100
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1,1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1,1)
    )
    x_new = np.c_[x0.ravel(), x1.ravel()] # 拉长再左右拼接，变为10000*2，一行是一个坐标点
    y_pred = model.predict(x_new)       # 对每一个坐标点进行预测
    zz = y_pred.reshape(x0.shape)       # 每一个对应坐标点的分类，100*100
    from matplotlib.colors import ListedColormap
    cus = ListedColormap(["#EF9A9A", "#FFF59D", "#90CAF9"]) # 设置颜色
    plt.contourf(x0, x1, zz, cmap = cus)                    # 绘制网格背景
    w = model.coef_[0]      # 线性核svm中的参数w
    b = model.intercept_[0] # 参数b
    x1 = np.linspace(axis[0], axis[1], 200)         # 分割超平面上点的横坐标
    upy = -w[0] * x1 / w[1] - b / w[1] + 1 / w[1]   # 将分隔超平面上的点纵坐标上移
    downy = -w[0] * x1 / w[1] - b / w[1] - 1 / w[1] # 将分隔超平面上的点纵坐标下移
    upindex = ((upy > axis[2]) & (upy < axis[3]))       # 选择用来画上支撑超平面的坐标点
    downindex = ((downy > axis[2]) & (downy < axis[3])) # 选择用来画下支撑超平面的坐标点
    plt.plot(x1[upindex], upy[upindex], "r")
    plt.plot(x1[downindex], downy[downindex], "g")
    
print('w:', s11.coef_[0])
print('b:', s11.intercept_[0])

plot_svc_decision_boundary(s11, axis = ([-3, 3, -3, 3]))
plt.scatter(x_standard[y == 0, 0], x_standard[y == 0, 1], color = "r")
plt.scatter(x_standard[y == 1, 0], x_standard[y == 1, 1], color = "g")
plt.tick_params(labelsize = 20)
plt.xlabel('x1', fontsize = 25)
plt.ylabel('x2', fontsize = 25)
plt.title('Sklearn result of SVM', fontsize = 30)
plt.show()











