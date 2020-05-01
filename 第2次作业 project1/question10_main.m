rand('seed', 1); % ������������ӣ���֤ÿ�γ�ʼ������ͬ
randn('seed',1); % ע����������������ĺ�����Ҫ����

load digits.mat
[n,d] = size(X);
nLabels = max(y); % ��ǩ��������
yExpanded = linearInd2Binary(y,nLabels); % ��1-10�ı�ǩ����Ϊone-hot����������Ϊ-1/1������0/1
t = size(Xvalid,1); % ��֤���ĵ�һ��ά�ȳ���
t2 = size(Xtest,1); % ���Լ��ĵ�һ��ά�ȳ���

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X); % ������׼����X�Լ�X�ľ�ֵ����
X = [ones(n,1) X]; % ��X�ĵ�һ��ǰ�����һ��ȫ1������
d = d + 1; % X���г��ȼ�һ

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid, mu, sigma);
Xvalid = [ones(t, 1) Xvalid];
Xtest = standardizeCols(Xtest, mu, sigma);
Xtest = [ones(t2, 1) Xtest];

% Choose network structure
nHidden1 = [10]; % ȫ����������ÿһ�����ز����Ԫ���������������Լ�����
kernel_size = 5;
nHidden2 = [1]; % CNN�����ز㣬��һ�������Ǿ�����������������ȫ�������ز����Ԫ����

% ����ȫ�����������е� 'w'
nParams = d * nHidden1(1);
for h = 2:length(nHidden1)
    nParams = nParams + nHidden1(h-1) * nHidden1(h); % ������˷��Ĳ����������ܲ�����
end
nParams = nParams + nHidden1(end) * nLabels; % ���һ������
w = randn(nParams, 1); % �����ʼ����������ʱ����һ��������

% ����CNN�еĲ�����
convParams = kernel_size^2 * nHidden2(1); % ��ʱ������bias
w1 = randn(convParams, 1);
if length(nHidden2) > 1
    connectParams = nHidden2(end) * nLabels; % ���һ������
    connectParams = connectParams + nHidden2(1) * d * nHidden2(2); % ������һ��ȫ����
else
    connectParams = nHidden2(1) * (d-1) * nLabels;
end
for h = 3:length(nHidden2)
    connectParams = connectParams + nHidden2(h-1) * nHidden2(h); % ������˷��Ĳ����������ܲ�����
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
    if mod(iter-1, round(maxIter/5)) == 0 % �ܹ�����֤���Ͻ���5����֤
        if conv == 1
            yhat = question10_Conv_Predict(w1, w2, Xvalid, kernel_size, nHidden2, nLabels); % �õ���֤��Ԥ����
        else
            yhat = MLPclassificationPredict(w, Xvalid, nHidden1, nLabels); % �õ���֤��Ԥ����
        end
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand * n); % ��1-n֮����ȡһ��������Ϊѡȡ�ļ����ݶȵ���������
    if conv == 1
        [f, g1, g2] = funObj(w1, w2, i);
        w1 = w1 - stepSize * g1; % ���о�����ݶ��½�����
        w2 = w2 - stepSize * g2; % ����ȫ���Ӳ��ݶ��½�����
    else
        [f, g] = funObj(w, i); % ����loss���ݶ�
        w = w - stepSize * g; % �����ݶ��½�����
    end
end
toc;

% Evaluate test error���ڲ��Լ��ϼ��������
if conv == 1
    yhat = question10_Conv_Predict(w1, w2, Xtest, kernel_size, nHidden2, nLabels); % �õ���֤��Ԥ����
else
    yhat = MLPclassificationPredict(w, Xtest, nHidden1, nLabels); % �õ���֤��Ԥ����
end
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);