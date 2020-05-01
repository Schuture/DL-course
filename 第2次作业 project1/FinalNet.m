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
kernel_size = 3;
nHidden = [8 128 128 64 32 32]; % 包含卷积的隐藏层，第一个参数是卷积核数量，后面的是全连接隐藏层的神经元数量

% 计数CNN中的参数量，并使用Xavier初始化
convParams = kernel_size^2 * nHidden(1); % 暂时不考虑bias
w1 = randn(convParams, 1) / (kernel_size * sqrt(nHidden(1)));
inputDims = [nHidden(1)*d, nHidden(2:length(nHidden)), 10];
connectParams = 0;
offset = 0;
for i=1:length(inputDims)-1
    connectParams = connectParams + inputDims(i)*inputDims(i+1);
    offset = offset + inputDims(i)*inputDims(i+1);
end
w2 = zeros(connectParams, 1);
offset = 0;
for i=1:length(inputDims)-1
    w2(offset+1:offset+inputDims(i)*inputDims(i+1),1) = randn(inputDims(i)*inputDims(i+1),1) / sqrt(inputDims(i));
    offset = offset + inputDims(i)*inputDims(i+1);
end

% Train with stochastic gradient
maxIter = 50000;
stepSize = 1e-4;
beta = 0.9;
val_errs = zeros(100);
funObj = @(w1,w2,i)question10_ConvNet(w1, w2, X(i, :), yExpanded(i, :), kernel_size, nHidden, nLabels);

tic;
diff_w1 = 0; % 用于记录momentum中的w(t) - w(t-1)
diff_w2 = 0;
j = 1;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/100)) == 0 % 总共在验证集上进行100次验证
        yhat = question10_Conv_Predict(w1, w2, Xvalid, kernel_size, nHidden, nLabels); % 得到验证集预测结果
        err = sum(yhat~=yvalid)/t;
        val_errs(j) = err;
        j = j + 1;
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, err);
    end
    
    i = ceil(rand * n); % 在1-n之间任取一个数，作为选取的计算梯度的样本索引
    [f, g1, g2] = funObj(w1, w2, i);
    if iter >= 2 % 从第二次迭代开始就可以记录参数差了
        diff_w1 =  - stepSize * g1 + beta * diff_w1;
        diff_w2 =  - stepSize * g2 + beta * diff_w2;
    end
    w1 = w1 - stepSize * g1 + beta * diff_w1; % 进行卷积层动量梯度下降更新
    w2 = w2 - stepSize * g2 + beta * diff_w2; % 进行全连接层动量梯度下降更新
end
toc;

% Evaluate test error，在测试集上计算错误率
iters = [0:500:49500];
figure(1);
plot(iters, val_errs);
title('Validation error of FinalNet')
xlabel('iteration');
ylabel('err rate');
yhat = question10_Conv_Predict(w1, w2, Xtest, kernel_size, nHidden, nLabels); % 得到验证集预测结果
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);