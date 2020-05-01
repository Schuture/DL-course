function [f,g] = question3_accelerated_MLP(w,X,y,nHidden,nLabels)

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
    
    relativeErr = yhat - y(i,:); % �в������
    f = f + sum(relativeErr.^2); % ���в�ƽ���ͼӵ�����ʧ��
    
    if nargout > 1
        err = 2 * relativeErr; % �������򴫲�

        % Output Weights
        %for c = 1:nLabels
        %    gOutput(:,c) = gOutput(:,c) + err(c) * fp{end}'; % ������ݶȼ���2�����˵����ڶ������
        %end
        % nHidden = [15]ʱ�����������ѭ���ĳ�����ľ���˷���13.13s -> 11.75s
        gOutput = gOutput + (err' * fp{end})'; 

        if length(nHidden) > 1 % �����������������������
            % Last Layer of Hidden Weights
            clear backprop
            %for c = 1:nLabels
            %    % �������򴫲����м��sech˫���������˫�����ҵĵ���
            %    backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
            %    gHidden{end} = gHidden{end} + fp{end-1}' * backprop(c, :);
            %end
            % ��nHidden=[150, 15]���������ѭ���ĳ�����ľ���˷���133.78s -> 108.02s
            backprop = err' .* (sech(ip{end}) .^ 2 .* outputWeights');
            gHidden{end} = gHidden{end} + repmat(fp{end-1}', 1, nLabels) * backprop;
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* sech(ip{h+1}) .^ 2;
                gHidden{h} = gHidden{h} + fp{h}' * backprop;
            end

            % Input Weights
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}) .^ 2;
            gInput = gInput + X(i,:)' * backprop;
        else % ֻ��һ������������
           % Input Weights
            %for c = 1:nLabels
            %    gInput = gInput + err(c) * X(i,:)' * (sech(ip{end}) .^ 2 .* outputWeights(:, c)');
            %end
            % nHidden=[15]ʱ���ĺ�11.75s -> 8.74s
            gInput = gInput + err .* X(i,:)' * (sech(ip{end}) .^ 2 .* outputWeights');
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
