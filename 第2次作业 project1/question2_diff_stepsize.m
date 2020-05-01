rand('seed', 0); % ������������ӣ���֤ÿ�γ�ʼ������ͬ
randn('seed',0); % ע����������������ĺ�����Ҫ����

load digits.mat % ����X, y
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
nHidden = [10]; % ÿһ�����ز����Ԫ���������������Լ�����

% Count number of parameters and initialize weights 'w'
nParams = d * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams + nHidden(h-1) * nHidden(h); % ������˷��Ĳ����������ܲ�����
end
nParams = nParams + nHidden(end) * nLabels; % ���һ������
w = randn(nParams, 1); % �����ʼ����������ʱ����һ��������

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3; % 1e-1 / 1e-2 / 1e-3 / 1e-4
mode = 3; % 123�ֱ�Ϊ����ѧϰ�ʡ�ָ���½�ѧϰ�ʡ�����
beta = 0.9;
val_errs = zeros(20);
funObj = @(w,i)MLPclassificationLoss(w, X(i, :), yExpanded(i, :), nHidden, nLabels);

tic;
j = 1; % ���ڼ�¼��֤�������ʵ�����ָ��
diff_w = 0; % ���ڼ�¼momentum�е�w(t) - w(t-1)
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/20)) == 0 % �ܹ�����֤���Ͻ���20����֤
        yhat = MLPclassificationPredict(w, Xvalid, nHidden, nLabels); % �õ���֤��Ԥ����
        err = sum(yhat~=yvalid)/t;
        val_errs(j) = err;
        j = j + 1;
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, err);
    end
    
    i = ceil(rand * n); % ��1-n֮����ȡһ��������Ϊѡȡ�ļ����ݶȵ���������
    [f, g] = funObj(w, i); % ����loss���ݶ�
    if mode == 1
        w = w - stepSize * g; % ���г���ѧϰ���ݶ��½�����
    end
    if mode == 2
        stepSize = stepSize * 0.1^(1/maxIter); % ָ��˥��ѧϰ��
        w = w - stepSize * g;
    end
    if mode == 3
        if iter >= 2 % �ӵڶ��ε�����ʼ�Ϳ��Լ�¼��������
            diff_w =  - stepSize * g + beta * diff_w;
        end
        w = w - stepSize * g + beta * diff_w; % �����ݶ��½�����
    end
end
toc;

% Evaluate test error�����������ʱ仯ͼ���ڲ��Լ��ϼ������մ�����
iters = [0:5000:95000];
figure(1);
plot(iters, val_errs);
title('Validation error of GD with momentum')
xlabel('iteration');
ylabel('err rate');
yhat = MLPclassificationPredict(w, Xtest, nHidden, nLabels);
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);