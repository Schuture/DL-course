rand('seed', 1); % 设置随机数种子，保证每次初始化都相同
randn('seed',1); % 注意两个生成随机数的函数都要设置

load digits.mat
[n,d] = size(X);
nLabels = max(y); % 标签的种类数
yExpanded = linearInd2Binary(y,nLabels); % 将1-10的标签编码为one-hot向量，但是为-1/1，不是0/1
t = size(Xvalid,1); % 验证集的第一个维度长度
t2 = size(Xtest,1); % 测试集的第一个维度长度

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X); % 经过标准化的X以及X的均值方差
X = [ones(n,1) X]; % 在X的第一列前面加上一个全1列向量
d = d + 1; % X的列长度加一

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid, mu, sigma);
Xvalid = [ones(t, 1) Xvalid];
Xtest = standardizeCols(Xtest, mu, sigma);
Xtest = [ones(t2, 1) Xtest];

% Choose network structure
nHidden1 = [10]; % 全连接神经网络每一层隐藏层的神经元数量，层数可以自己增加
kernel_size = 5;
nHidden2 = [1]; % CNN的隐藏层，第一个参数是卷积核数量，后面的是全连接隐藏层的神经元数量

% 计数全连接神经网络中的 'w'
nParams = d * nHidden1(1);
for h = 2:length(nHidden1)
    nParams = nParams + nHidden1(h-1) * nHidden1(h); % 按矩阵乘法的参数量计算总参数量
end
nParams = nParams + nHidden1(end) * nLabels; % 最后一层分类层
w = randn(nParams, 1); % 随机初始化参数，暂时存在一个向量中

% 计数CNN中的参数量
convParams = kernel_size^2 * nHidden2(1); % 暂时不考虑bias
w1 = randn(convParams, 1);
if length(nHidden2) > 1
    connectParams = nHidden2(end) * nLabels; % 最后一层分类层
    connectParams = connectParams + nHidden2(1) * d * nHidden2(2); % 卷积后第一层全连接
else
    connectParams = nHidden2(1) * (d-1) * nLabels;
end
for h = 3:length(nHidden2)
    connectParams = connectParams + nHidden2(h-1) * nHidden2(h); % 按矩阵乘法的参数量计算总参数量
end
w2 = randn(connectParams, 1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
conv = 0;
if conv == 1
    funObj = @(w1,w2,i)question10_ConvNet(w1, w2, X(i, :), yExpanded(i, :), kernel_size, nHidden2, nLabels);
else
    funObj = @(w,i)MLPclassificationLoss(w, X(i, :), yExpanded(i, :), nHidden1, nLabels);
end
tic;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/5)) == 0 % 总共在验证集上进行5次验证
        if conv == 1
            yhat = question10_Conv_Predict(w1, w2, Xvalid, kernel_size, nHidden2, nLabels); % 得到验证集预测结果
        else
            yhat = MLPclassificationPredict(w, Xvalid, nHidden1, nLabels); % 得到验证集预测结果
        end
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand * n); % 在1-n之间任取一个数，作为选取的计算梯度的样本索引
    if conv == 1
        [f, g1, g2] = funObj(w1, w2, i);
        w1 = w1 - stepSize * g1; % 进行卷积层梯度下降更新
        w2 = w2 - stepSize * g2; % 进行全连接层梯度下降更新
    else
        [f, g] = funObj(w, i); % 计算loss与梯度
        w = w - stepSize * g; % 进行梯度下降更新
    end
end
toc;

% Evaluate test error，在测试集上计算错误率
if conv == 1
    yhat = question10_Conv_Predict(w1, w2, Xtest, kernel_size, nHidden2, nLabels); % 得到验证集预测结果
else
    yhat = MLPclassificationPredict(w, Xtest, nHidden1, nLabels); % 得到验证集预测结果
end
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);