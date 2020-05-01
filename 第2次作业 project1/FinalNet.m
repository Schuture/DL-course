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
kernel_size = 3;
nHidden = [8 128 128 64 32 32]; % ������������ز㣬��һ�������Ǿ�����������������ȫ�������ز����Ԫ����

% ����CNN�еĲ���������ʹ��Xavier��ʼ��
convParams = kernel_size^2 * nHidden(1); % ��ʱ������bias
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
diff_w1 = 0; % ���ڼ�¼momentum�е�w(t) - w(t-1)
diff_w2 = 0;
j = 1;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/100)) == 0 % �ܹ�����֤���Ͻ���100����֤
        yhat = question10_Conv_Predict(w1, w2, Xvalid, kernel_size, nHidden, nLabels); % �õ���֤��Ԥ����
        err = sum(yhat~=yvalid)/t;
        val_errs(j) = err;
        j = j + 1;
        fprintf('Training iteration = %d, validation error = %f\n', iter-1, err);
    end
    
    i = ceil(rand * n); % ��1-n֮����ȡһ��������Ϊѡȡ�ļ����ݶȵ���������
    [f, g1, g2] = funObj(w1, w2, i);
    if iter >= 2 % �ӵڶ��ε�����ʼ�Ϳ��Լ�¼��������
        diff_w1 =  - stepSize * g1 + beta * diff_w1;
        diff_w2 =  - stepSize * g2 + beta * diff_w2;
    end
    w1 = w1 - stepSize * g1 + beta * diff_w1; % ���о���㶯���ݶ��½�����
    w2 = w2 - stepSize * g2 + beta * diff_w2; % ����ȫ���Ӳ㶯���ݶ��½�����
end
toc;

% Evaluate test error���ڲ��Լ��ϼ��������
iters = [0:500:49500];
figure(1);
plot(iters, val_errs);
title('Validation error of FinalNet')
xlabel('iteration');
ylabel('err rate');
yhat = question10_Conv_Predict(w1, w2, Xtest, kernel_size, nHidden, nLabels); % �õ���֤��Ԥ����
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);