[Y_train, X_train] = libsvmread('train.txt');
[Y_test, X_test] = libsvmread('test.txt');
class_num = length(unique(Y_train));
model = cell(class_num, 1);
kernel_type = 'rbf';

tic;
for label = 1 : class_num
    Y_label = (Y_train == label-1);
    switch kernel_type
        case 'rbf'
            model{label} = svmtrain(double(Y_label), X_train, '-t 2 -g 500 -b 1');
        case 'linear'
            model{label} = svmtrain(double(Y_label), X_train, '-t 0 -b 1');
        case 'poly'
            model{label} = svmtrain(double(Y_label), X_train, '-t 1 -b 1');
    end
end
toc;

Y_prob = zeros(size(Y_test, 1), class_num);
for label = 1 : class_num
    [~, ~, dv] = svmpredict(Y_test, X_test, model{label}, '-b 1');
    if model{label}.Label(1) == 0
        Y_prob(:, label) = dv(:, 2);
    else
        Y_prob(:, label) = dv(:, 1);
    end
end

[~, Y_pred] = max(Y_prob, [], 2);
Y_pred = Y_pred - 1;
disp(['The accuracy is: ', num2str(sum(Y_pred == Y_test)/size(Y_test, 1))]);
