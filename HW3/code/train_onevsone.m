[Y_train, X_train] = libsvmread('train.txt');
[Y_test, X_test] = libsvmread('test.txt');
class_num = length(unique(Y_train));
model = cell(class_num, class_num);
kernel_type = 'poly';
tic;
for i = 1 : class_num
    for j = 1 : class_num
        if i > j
            pos_index = (Y_train == i-1);
            neg_index = (Y_train == j-1);
            Y_label = Y_train(pos_index | neg_index, :);
            X_fea = X_train(pos_index | neg_index, :);
            switch kernel_type
                case 'rbf'
                    model{i, j} = svmtrain(Y_label, X_fea, '-t 2 -g 500 -b 1');
                case 'linear'
                    model{i, j} = svmtrain(Y_label, X_fea, '-t 0 -b 1');
                case 'poly'
                    model{i, j} = svmtrain(Y_label, X_fea, '-t 1 -b 1');                    
            end
        end
    end
end
toc;

Y_prob = zeros(size(Y_test, 1), class_num);
for i = 1 : class_num
    for j = 1 : class_num
        if i > j
            [~, ~, dv] = svmpredict(Y_test, X_test, model{i, j}, '-b 1');
            if model{i, j}.Label(1) == i-1
                Y_prob(:, i) = Y_prob(:, i) + dv(:, 1);
                Y_prob(:, j) = Y_prob(:, j) + dv(:, 2);
            else
                Y_prob(:, i) = Y_prob(:, i) + dv(:, 2);
                Y_prob(:, j) = Y_prob(:, j) + dv(:, 1);
            end
        end
    end
end

[~, Y_pred] = max(Y_prob, [], 2);
Y_pred = Y_pred - 1;
disp(['The accuracy is: ', num2str(sum(Y_pred == Y_test)/size(Y_test, 1))]);
