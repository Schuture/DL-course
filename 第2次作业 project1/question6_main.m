rand('seed', 0); % 设置随机数种子，保证每次初始化都相同
randn('seed',0); % 注意两个生成随机数的函数都要设置

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
nHidden = [15 10 10 10]; % 每一层隐藏层的神经元数量，层数可以自己增加

% Count number of parameters and initialize weights 'w'
nParams = d * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams + nHidden(h-1) * nHidden(h); % 按矩阵乘法的参数量计算总参数量
end
nParams = nParams + nHidden(end) * nLabels; % 最后一层分类层
w = randn(nParams, 1); % 随机初始化参数w，暂时存在一个向量中

% 计数并初始化偏置项 'b'
nb = sum(nHidden) + nLabels;
b = randn(nb, 1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
bias = 0; % 修改这里为1/0可以选择是有偏置项还是无偏置项
if bias == 1
    funObj = @(w,b,i)question6_MLP_with_bias(w, b, X(i, :), yExpanded(i, :), nHidden, nLabels);
else
    funObj = @(w,i)MLPclassificationLoss(w, X(i, :), yExpanded(i, :), nHidden, nLabels);
end
tic;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/5)) == 0 % 总共在验证集上进行5次验证
        if bias == 1
            yhat = question6_Predict_with_bias(w, b, Xvalid, nHidden, nLabels); % 得到验证集预测结果
        else
            yhat = MLPclassificationPredict(w, Xvalid, nHidden, nLabels); % 得到验证集预测结果
        end
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand * n); % 在1-n之间任取一个数，作为选取的计算梯度的样本索引
    if bias == 1
        [f, gw, gb] = funObj(w,b,i); % 计算loss与梯度，别忘了加上参数b
        w = w - stepSize * gw; % 进行梯度下降更新
        b = b - stepSize * gb;
    else
        [f, g] = funObj(w, i);
        w = w - stepSize * g;
    end
end
toc;

% Evaluate test error，在测试集上计算错误率
if bias == 1
    yhat = question6_Predict_with_bias(w, b, Xtest, nHidden, nLabels);
else
    yhat = MLPclassificationPredict(w, Xtest, nHidden, nLabels);
end
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);