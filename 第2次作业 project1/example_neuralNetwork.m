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
nHidden = [15 10 10 10 10]; % 每一层隐藏层的神经元数量，层数可以自己增加

% Count number of parameters and initialize weights 'w'
nParams = d * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams + nHidden(h-1) * nHidden(h); % 按矩阵乘法的参数量计算总参数量
end
nParams = nParams + nHidden(end) * nLabels; % 最后一层分类层
w = randn(nParams, 1); % 随机初始化参数，暂时存在一个向量中

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
%funObj = @(w,i)question3_accelerated_MLP(w, X(i, :), yExpanded(i, :), nHidden, nLabels);
funObj = @(w,i)MLPclassificationLoss(w, X(i, :), yExpanded(i, :), nHidden, nLabels);
tic;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/5)) == 0 % 总共在验证集上进行5次验证
        yhat = MLPclassificationPredict(w, Xvalid, nHidden, nLabels); % 得到验证集预测结果
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand * n); % 在1-n之间任取一个数，作为选取的计算梯度的样本索引
    [f, g] = funObj(w, i); % 计算loss与梯度
    w = w - stepSize * g; % 进行梯度下降更新
end
toc;

% Evaluate test error，在测试集上计算错误率
yhat = MLPclassificationPredict(w, Xtest, nHidden, nLabels);
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);