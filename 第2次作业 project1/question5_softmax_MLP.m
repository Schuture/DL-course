function [f,g] = question5_softmax_MLP(w,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X); % 样本数量，变量数量

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)), nVars, nHidden(1)); % 第一层参数
offset = nVars * nHidden(1); % 一个指针，表明下一层的参数从哪里开始取
for h = 2:length(nHidden)
  % 将权重向量w的一部分截取出来并转化为这一层的矩阵形式
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h); % 每截取一层的参数，更新指针
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels); % 最后一层权重，向量
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % 最后一层权重，矩阵

f = 0; % 初始化损失函数值
if nargout > 1
    gInput = zeros(size(inputWeights)); % 初始化输入层的梯度
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); % 初始化隐藏层的梯度
    end
    gOutput = zeros(size(outputWeights)); % 初始化输出层的梯度
end

% Compute Output
for i = 1:nInstances % 遍历每一个样本
    ip{1} = X(i, :) * inputWeights; % 数据经过第一层权重层
    fp{1} = tanh(ip{1}); % 经过激活函数
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1}; % 逐一经过隐藏层
        fp{h} = tanh(ip{h}); % 经过激活函数
    end
    yhat = fp{end} * outputWeights; % 经过最后一层权重
    p = softmax(yhat')'; % 经过softmax函数

    f = f - log(p(y(i))); % 负对数似然损失(CELoss)
    
    if nargout > 1
        Sj = -p;
        Sj(y(i)) = Sj(y(i)) + 1;
        err = -p .* Sj; % 最后一层的误差项，反向传播

        % Output Weights
        for c = 1:nLabels
            gOutput(:,c) = gOutput(:,c) + err(c) * fp{end}'; % 输出层梯度加上2倍误差乘倒数第二层输出
        end

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            for c = 1:nLabels
                % 用来反向传播的中间项，sech双曲正割函数是双曲余弦的倒数
                backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
                gHidden{end} = gHidden{end} + fp{end-1}' * backprop(c, :);
            end
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* sech(ip{h+1}) .^ 2;
                gHidden{h} = gHidden{h} + fp{h}' * backprop;
            end

            % Input Weights
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}) .^ 2;
            gInput = gInput + X(i,:)' * backprop;
        else
           % Input Weights
            for c = 1:nLabels
                gInput = gInput + err(c) * X(i,:)' * (sech(ip{end}) .^ 2 .* outputWeights(:, c)');
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1 % 如果包含隐藏层之间的参数，就一层一层将对应梯度值放进g中
    g = zeros(size(w)); % 初始化返回的梯度g
    g(1:nVars*nHidden(1)) = gInput(:); % 输入层梯度
    offset = nVars * nHidden(1); % 一个指针，表明下一层的梯度放到哪里
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset + nHidden(h-1) * nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:); % 输出层梯度
end
end

function [y] = softmax(x) % 对列向量进行softmax变换
for i = 1:length(x)
    x(i) = exp(x(i));
end
y = x / sum(x);
end