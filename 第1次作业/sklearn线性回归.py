from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# 第一步 生成数据
X = np.random.rand(20, 1) * 10 # sklearn线性模型只接受二维数据
y = 3 * X + (5 + np.random.randn(20, 1))

# 第二步 定义模型
model = linear_model.LinearRegression()

# 第三步 拟合数据
model.fit(X, y)

# 第四步 预测（以下两种方法都可以）
# y_pred = k * X + b
y_pred = model.predict(X)

# 第四步 展示拟合结果
b = model.intercept_.item()
k = model.coef_.item()
print('截距为：', b)  #截距
print('斜率为：', k)  #线性模型的系数

plt.figure(figsize = (12,12))
plt.scatter(X.reshape(-1), y.reshape(-1))
plt.plot(X.reshape(-1), y_pred.reshape(-1), 'r-', lw=5)
plt.tick_params(labelsize=20)
plt.xlabel('X', fontsize = 25)
plt.ylabel('y', fontsize = 25)
plt.xlim(0, 10)
plt.ylim(8, 35)
plt.title('Sklearn result of linear regression', fontsize = 30)
plt.show()



