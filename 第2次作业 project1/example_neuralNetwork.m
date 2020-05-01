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
nHidden = [15 10 10 10 10]; % ÿһ�����ز����Ԫ���������������Լ�����

% Count number of parameters and initialize weights 'w'
nParams = d * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams + nHidden(h-1) * nHidden(h); % ������˷��Ĳ����������ܲ�����
end
nParams = nParams + nHidden(end) * nLabels; % ���һ������
w = randn(nParams, 1); % �����ʼ����������ʱ����һ��������

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
%funObj = @(w,i)question3_accelerated_MLP(w, X(i, :), yExpanded(i, :), nHidden, nLabels);
funObj = @(w,i)MLPclassificationLoss(w, X(i, :), yExpanded(i, :), nHidden, nLabels);
tic;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/5)) == 0 % �ܹ�����֤���Ͻ���5����֤
        yhat = MLPclassificationPredict(w, Xvalid, nHidden, nLabels); % �õ���֤��Ԥ����
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand * n); % ��1-n֮����ȡһ��������Ϊѡȡ�ļ����ݶȵ���������
    [f, g] = funObj(w, i); % ����loss���ݶ�
    w = w - stepSize * g; % �����ݶ��½�����
end
toc;

% Evaluate test error���ڲ��Լ��ϼ��������
yhat = MLPclassificationPredict(w, Xtest, nHidden, nLabels);
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);