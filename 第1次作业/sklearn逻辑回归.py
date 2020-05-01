from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# 第一步 生成数据
sample_nums = 100
mean = np.array([1.7, 1.7]) # 一类样本均值所处的位置，例如[1.7, 1.7]
cov = [[1,0],[0,1]]
bias = 0 # 数据样本整体平移
x0 = np.random.multivariate_normal(mean, cov, sample_nums) + bias
y0 = np.zeros((sample_nums,1))
x1 = np.random.multivariate_normal(-mean, cov, sample_nums) + bias
y1 = np.ones((sample_nums,1))

X = np.vstack((x0, x1))
y = np.vstack((y0, y1)).reshape(-1)

# 第二步 定义模型
model = LogisticRegression(solver='liblinear')

# 第三步 拟合数据
model.fit(X, y)

# 第四步 预测
y_pred = model.predict(X)

# 第五步 展示拟合结果
print('测试准确率：', accuracy_score(y, y_pred))

weights = np.column_stack((model.intercept_, model.coef_)).transpose() # 阶矩，斜率

fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(111)
ax.scatter(x0[:,0], x0[:,1], s=30, c='red', marker='s')
ax.scatter(x1[:,0], x1[:,1], s=30, c='green')
x_ = np.arange(-3.0, 3.0, 0.1)
y_ = (-weights[0] - weights[1] * x_) / weights[2]
ax.plot(x_, y_)
plt.tick_params(labelsize=20)
plt.xlabel('x1', fontsize = 25)
plt.ylabel('x2', fontsize = 25)
plt.title('Sklearn result of logistic regression', fontsize = 30)
plt.show()










