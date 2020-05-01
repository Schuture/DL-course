function [y] = question6_Predict_with_bias(w, b, X, nHidden, nLabels)
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars * nHidden(1)), nVars, nHidden(1)); % 第一层的权重，矩阵形式
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

% Compute Output
for i = 1:nInstances % 遍历每一个样本
    ip{1} = X(i, :) * inputWeights + inputBias'; % 数据经过第一层权重层
    fp{1} = tanh(ip{1}); % 经过激活函数
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1} + hiddenBias{h-1}'; % 逐一经过隐藏层
        fp{h} = tanh(ip{h}); % 经过激活函数
    end
    y(i,:) = fp{end} * outputWeights;
end
[v,y] = max(y,[],2); % 最终y取的是每一个样本预测向量的最大值位置
end
