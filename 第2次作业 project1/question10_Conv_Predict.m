function [y] = question10_Conv_Predict(w1, w2, X, kernel_size, nHidden, nLabels)

X = X(:,2:257);
[nInstances,nVars] = size(X);
height = round(sqrt(nVars));
width = nVars / height;

% Form Weights
nkernels = nHidden(1);
inputWeights = reshape(w1, nkernels, kernel_size^2); % 卷积层参数
nHidden(1) = nVars * nkernels; % 特征图展开后的长度
offset = 0; % 一个指针，表明下一层的参数从哪里开始取
for h = 2:length(nHidden)
  % 将权重向量w的一部分截取出来并转化为这一层的矩阵形式
  hiddenWeights{h-1} = reshape(w2(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h); % 每截取一层的参数，更新指针
end
outputWeights = w2(offset+1:offset+nHidden(end)*nLabels); % 最后一层权重，向量
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % 最后一层权重，矩阵

% Compute Output
for i = 1:nInstances % 遍历每一个样本
    image = reshape(X(i,:), height, width);
    for j = 1:nkernels % 卷积操作，nkernels个卷积核对应nkernels个特征图
        kernel{j} = reshape(inputWeights(j,:), kernel_size, kernel_size);
        featmap{j} = conv2(image, kernel{j}, 'same');
        ip{1}((j-1)*nVars+1:j*nVars) = reshape(featmap{j}, 1, nVars); % 将特征图拉直
    end
    fp{1} = tanh(ip{1}); % 经过激活函数
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1}; % 逐一经过隐藏层
        fp{h} = tanh(ip{h}); % 经过激活函数
    end
    y(i,:) = fp{end} * outputWeights;
end

[v,y] = max(y,[],2); % 最终y取的是每一个样本预测向量的最大值位置
end
