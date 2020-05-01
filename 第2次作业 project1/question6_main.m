rand('seed', 0); % ������������ӣ���֤ÿ�γ�ʼ������ͬ
randn('seed',0); % ע����������������ĺ�����Ҫ����

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
nHidden = [15 10 10 10]; % ÿһ�����ز����Ԫ���������������Լ�����

% Count number of parameters and initialize weights 'w'
nParams = d * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams + nHidden(h-1) * nHidden(h); % ������˷��Ĳ����������ܲ�����
end
nParams = nParams + nHidden(end) * nLabels; % ���һ������
w = randn(nParams, 1); % �����ʼ������w����ʱ����һ��������

% ��������ʼ��ƫ���� 'b'
nb = sum(nHidden) + nLabels;
b = randn(nb, 1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
bias = 0; % �޸�����Ϊ1/0����ѡ������ƫ�������ƫ����
if bias == 1
    funObj = @(w,b,i)question6_MLP_with_bias(w, b, X(i, :), yExpanded(i, :), nHidden, nLabels);
else
    funObj = @(w,i)MLPclassificationLoss(w, X(i, :), yExpanded(i, :), nHidden, nLabels);
end
tic;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/5)) == 0 % �ܹ�����֤���Ͻ���5����֤
        if bias == 1
            yhat = question6_Predict_with_bias(w, b, Xvalid, nHidden, nLabels); % �õ���֤��Ԥ����
        else
            yhat = MLPclassificationPredict(w, Xvalid, nHidden, nLabels); % �õ���֤��Ԥ����
        end
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand * n); % ��1-n֮����ȡһ��������Ϊѡȡ�ļ����ݶȵ���������
    if bias == 1
        [f, gw, gb] = funObj(w,b,i); % ����loss���ݶȣ������˼��ϲ���b
        w = w - stepSize * gw; % �����ݶ��½�����
        b = b - stepSize * gb;
    else
        [f, g] = funObj(w, i);
        w = w - stepSize * g;
    end
end
toc;

% Evaluate test error���ڲ��Լ��ϼ��������
if bias == 1
    yhat = question6_Predict_with_bias(w, b, Xtest, nHidden, nLabels);
else
    yhat = MLPclassificationPredict(w, Xtest, nHidden, nLabels);
end
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);