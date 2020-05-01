function [f,g] = question5_softmax_MLP(w,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X); % ������������������

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)), nVars, nHidden(1)); % ��һ�����
offset = nVars * nHidden(1); % һ��ָ�룬������һ��Ĳ��������￪ʼȡ
for h = 2:length(nHidden)
  % ��Ȩ������w��һ���ֽ�ȡ������ת��Ϊ��һ��ľ�����ʽ
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h); % ÿ��ȡһ��Ĳ���������ָ��
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels); % ���һ��Ȩ�أ�����
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % ���һ��Ȩ�أ�����

f = 0; % ��ʼ����ʧ����ֵ
if nargout > 1
    gInput = zeros(size(inputWeights)); % ��ʼ���������ݶ�
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); % ��ʼ�����ز���ݶ�
    end
    gOutput = zeros(size(outputWeights)); % ��ʼ���������ݶ�
end

% Compute Output
for i = 1:nInstances % ����ÿһ������
    ip{1} = X(i, :) * inputWeights; % ���ݾ�����һ��Ȩ�ز�
    fp{1} = tanh(ip{1}); % ���������
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1}; % ��һ�������ز�
        fp{h} = tanh(ip{h}); % ���������
    end
    yhat = fp{end} * outputWeights; % �������һ��Ȩ��
    p = softmax(yhat')'; % ����softmax����

    f = f - log(p(y(i))); % ��������Ȼ��ʧ(CELoss)
    
    if nargout > 1
        Sj = -p;
        Sj(y(i)) = Sj(y(i)) + 1;
        err = -p .* Sj; % ���һ����������򴫲�

        % Output Weights
        for c = 1:nLabels
            gOutput(:,c) = gOutput(:,c) + err(c) * fp{end}'; % ������ݶȼ���2�����˵����ڶ������
        end

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            for c = 1:nLabels
                % �������򴫲����м��sech˫���������˫�����ҵĵ���
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
if nargout > 1 % ����������ز�֮��Ĳ�������һ��һ�㽫��Ӧ�ݶ�ֵ�Ž�g��
    g = zeros(size(w)); % ��ʼ�����ص��ݶ�g
    g(1:nVars*nHidden(1)) = gInput(:); % ������ݶ�
    offset = nVars * nHidden(1); % һ��ָ�룬������һ����ݶȷŵ�����
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset + nHidden(h-1) * nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:); % ������ݶ�
end
end

function [y] = softmax(x) % ������������softmax�任
for i = 1:length(x)
    x(i) = exp(x(i));
end
y = x / sum(x);
end