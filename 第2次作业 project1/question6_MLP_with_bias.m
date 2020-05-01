function [f,gw,gb] = question6_MLP_with_bias(w,b,X,y,nHidden,nLabels)

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

% Form bias
inputBias = b(1:nHidden(1));
offset = nHidden(1);
for h = 2:length(nHidden)
    hiddenBias{h-1} = b(offset+1:offset+nHidden(h));
    offset = offset + nHidden(h);
end
outputBias = b(offset+1:offset+nLabels);

f = 0; % 初始化损失函数值
if nargout > 1
    gwInput = zeros(size(inputWeights)); % 初始化输入层w的梯度
    for h = 2:length(nHidden)
       gwHidden{h-1} = zeros(size(hiddenWeights{h-1})); % 初始化隐藏层w的梯度
    end
    gwOutput = zeros(size(outputWeights)); % 初始化输出层w的梯度
    
    gbInput = zeros(size(inputBias)); % 初始化输入层b的梯度
    for h = 2:length(nHidden)
       gbHidden{h-1} = zeros(size(hiddenBias{h-1})); % 初始化隐藏层b的梯度
    end
    gbOutput = zeros(size(outputBias)); % 初始化输出层b的梯度
end

% Compute Output
for i = 1:nInstances % 遍历每一个样本
    ip{1} = X(i, :) * inputWeights + inputBias'; % 数据经过第一层权重层
    fp{1} = tanh(ip{1}); % 经过激活函数
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1} + hiddenBias{h-1}'; % 逐一经过隐藏层
        fp{h} = tanh(ip{h}); % 经过激活函数
    end
    yhat = fp{end} * outputWeights + outputBias'; % 经过最后一层权重
    
    relativeErr = yhat - y(i,:); % 残差
    f = f + sum(relativeErr.^2); % 将残差平方和加到总损失上
    
    if nargout > 1
        err = 2 * relativeErr; % 误差项，反向传播

        % Output Weights
        for c = 1:nLabels
            gwOutput(:,c) = gwOutput(:,c) + err(c) * fp{end}'; % 输出层梯度加上2倍误差乘倒数第二层输出
            gbOutput(c) = gbOutput(c) + err(c);
        end

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            for c = 1:nLabels
                % 用来反向传播的中间项，sech双曲正割函数是双曲余弦的倒数
                backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
                gwHidden{end} = gwHidden{end} + fp{end-1}' * backprop(c, :);
                gbHidden{end} = gbHidden{end} + backprop(c, :)';
            end
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* sech(ip{h+1}) .^ 2;
                gwHidden{h} = gwHidden{h} + fp{h}' * backprop;
                gbHidden{h} = gbHidden{h} + backprop';
            end

            % Input Weights
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}) .^ 2;
            gwInput = gwInput + X(i,:)' * backprop;
            gbInput = gbInput + backprop';
        else
            % Input Weights
            for c = 1:nLabels
                backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
                gwInput = gwInput + X(i,:)' * backprop(c,:);
                gbInput = gbInput + backprop(c, :)';
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1 % 如果包含隐藏层之间的参数，就一层一层将对应梯度值放进g中
    gw = zeros(size(w)); % 初始化返回的w的梯度gw
    gw(1:nVars*nHidden(1)) = gwInput(:); % 输入层w梯度
    offset = nVars * nHidden(1); % 一个指针，表明下一层的梯度放到哪里
    for h = 2:length(nHidden)
        gw(offset+1:offset+nHidden(h-1)*nHidden(h)) = gwHidden{h-1};
        offset = offset + nHidden(h-1) * nHidden(h);
    end
    gw(offset+1:offset+nHidden(end)*nLabels) = gwOutput(:); % 输出层w梯度
    
    gb = zeros(size(b)); % 初始化返回的b的梯度gb
    gb(1:nHidden(1)) = gbInput(:); % 输入层b梯度
    offset = nHidden(1); % 一个指针，表明下一层的梯度放到哪里
    for h = 2:length(nHidden)
        gb(offset+1:offset+nHidden(h)) = gbHidden{h-1};
        offset = offset + nHidden(h);
    end
    gb(offset+1:offset+nLabels) = gbOutput(:); % 输出层b梯度
end
end
