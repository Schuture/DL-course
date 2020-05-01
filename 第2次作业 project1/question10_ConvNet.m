function [f,g1,g2] = question10_ConvNet(w1,w2,X,y,kernel_size,nHidden,nLabels)

X = X(:,2:257); % 舍弃掉第一列偏置项
[nInstances,nVars] = size(X); % 样本数量，变量数量
height = round(sqrt(nVars));
width = nVars / height;

% Form Weights
nkernels = nHidden(1);
inputWeights = reshape(w1(1:kernel_size^2*nkernels), nkernels, kernel_size^2); % 卷积层参数
nHidden(1) = nVars * nkernels; % 特征图展开后的长度
offset = 0; % 一个指针，表明下一层的参数从哪里开始取
for h = 2:length(nHidden)
  % 将权重向量w的一部分截取出来并转化为这一层的矩阵形式
  hiddenWeights{h-1} = reshape(w2(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h); % 每截取一层的参数，更新指针
end
outputWeights = w2(offset+1:offset+nHidden(end)*nLabels); % 最后一层权重，向量
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % 最后一层权重，矩阵

f = 0; % 初始化损失函数值
if nargout > 1 % 初始化梯度
    gInput = zeros(size(w1)); % 初始化输入层的梯度
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); % 初始化隐藏层的梯度
    end
    gOutput = zeros(size(outputWeights)); % 初始化输出层的梯度
end

% Compute Output
for i = 1:nInstances % 遍历每一个样本
    image = reshape(X(i,:), height, width);
    image = padarray(image, [floor(kernel_size/2), floor(kernel_size/2)]);
    for j = 1:nkernels % 卷积操作，nkernels个卷积核对应nkernels个特征图
        kernel{j} = reshape(inputWeights(j,:), kernel_size, kernel_size);
        featmap{j} = conv2(image, kernel{j}, 'valid');
        ip{1}((j-1)*nVars+1:j*nVars) = reshape(featmap{j}, 1, nVars); % 将特征图拉直
    end
    fp{1} = tanh(ip{1}); % 经过激活函数
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1}; % 逐一经过隐藏层
        fp{h} = tanh(ip{h}); % 经过激活函数
    end
    yhat = fp{end} * outputWeights; % 经过最后一层权重
    
    relativeErr = yhat - y(i,:); % 残差
    f = f + sum(relativeErr.^2); % 将残差平方和加到总损失上
    
    if nargout > 1
        err = 2 * relativeErr; % 误差项，反向传播

        % Output Weights
        gOutput = gOutput + (err' * fp{end})'; 

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            % clear backprop % 去掉这一行以后速度X2.5
            backprop = err' .* (sech(ip{end}) .^ 2 .* outputWeights');
            gHidden{end} = gHidden{end} + repmat(fp{end-1}', 1, nLabels) * backprop;
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* sech(ip{h+1}) .^ 2;
                gHidden{h} = gHidden{h} + fp{h}' * backprop;
            end

            % Input Weights，其实到了这一步不需要进行卷积误差的反向传播，因为只有一层卷积
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}) .^ 2; % 1 * 768
            for j = 1:nkernels
                delta{j} = reshape(backprop((j-1)*nVars+1:j*nVars), height, width);
                grad = conv2(image, imrotate(delta{j}, 180, 'bilinear'), 'valid');
                gInput((j-1)*kernel_size^2+1:j*kernel_size^2) = reshape(grad, kernel_size^2, 1);
            end
        else
            % Input Weights
            backprop = err * (sech(ip{end}) .^ 2 .* outputWeights');
            for j = 1:nkernels
                delta{j} = reshape(backprop((j-1)*nVars+1:j*nVars), height, width);
                grad = conv2(image, imrotate(delta{j}, 180, 'bilinear'), 'valid');
                gInput((j-1)*kernel_size^2+1:j*kernel_size^2) = reshape(grad, kernel_size^2, 1);
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1 % 如果包含隐藏层之间的参数，就一层一层将对应梯度值放进g中
    g1 = gInput; % 输入层梯度
    g2 = zeros(size(w2));
    offset = 0; % 一个指针，表明下一层的梯度放到哪里
    for h = 2:length(nHidden)
        g2(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset + nHidden(h-1) * nHidden(h);
    end
    g2(offset+1:offset+nHidden(end)*nLabels) = gOutput(:); % 输出层梯度
end
