
%%四个参数依次是a1,a2,eta1,eta2
function BP_online = BP_online(p1,p2,p3,p4)
    data_train = load('two_spiral_train.txt');
    data_test = load('two_spiral_test.txt');



    %%------------------------STEP1 INITIALIZE-----------------------------
    %%hidden层神经元设定为10个
    num_of_hidden_neurons = 10; 
    %%每行的三个元素依次是第一个输入的u和第二个输入的u和偏置值1的权值，一共10行表示有十个hidden层神经元
    u1 = rand(num_of_hidden_neurons,3).*2-1;
    delta_u1_temp = zeros(num_of_hidden_neurons,3); %%存储temp delta u
    %%每行的2个元素依次是第一个输入的v和第二个输入的v，一共10行表示有十个hidden层神经元
    v1 = rand(num_of_hidden_neurons,2).*2-1;
    delta_v1_temp = zeros(num_of_hidden_neurons,2);

    %%11个权值分别是10个hidden层神经元到输出层的u权值以及偏置值1的权值
    u2 = rand(num_of_hidden_neurons+1,1).*2-1;
    delta_u2_temp = zeros(num_of_hidden_neurons+1,1);
    %%10个权值分别是10个hidden层神经元到输出层的v权值
    v2 = rand(num_of_hidden_neurons,1).*2-1;
    delta_v2_temp = zeros(num_of_hidden_neurons,1);

    %%初始化参数
    a1 = p1;
    a2 = p2;
    eta1 = p3;
    eta2 = p4;


    
    %%------------------------STEP2 TRAINING-------------------------------
    %%循环训练
    t1 = clock;
    for i=1:size(data_train,1)
        raw_input1 = data_train(i,1);
        raw_input2 = data_train(i,2);
        raw_output = data_train(i,3);

        %%顺序执行
        %%存储hidden神经元的实际输入(sigmoid函数的自变量)
        hidden_input = zeros(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_input(j,1) = input_of_hidden(raw_input1,raw_input2,u1(j,1),u1(j,2),u1(j,3),v1(j,1),v1(j,2));
        end 
        %%存储hidden神经元的输出，为中间十个神经元的输入的sigmoid函数输出
        hidden_output = zeros(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_output(j,1) = logsig(hidden_input(j,1));
        end

        %%存储output神经元的输入
        output_input = input_of_output(u2,v2,hidden_output);
        %%存储output神经元的输出 完成feed forward，接下来开始back propagation
        output_output = logsig(output_input);

        %%back propagation 1: output layer
        %%误差
        err = raw_output - output_output;

        delta_outputlayer = err * exp(-1 * output_input) * output_output^2;
        for j=1:num_of_hidden_neurons
            delta_u2_temp(j,1) = a1*delta_u2_temp(j,1) + eta1 * delta_outputlayer * hidden_output(j,1)^2;
            delta_v2_temp(j,1) = a2*delta_v2_temp(j,1) + eta2 * delta_outputlayer * hidden_output(j,1);
        end
        %%偏置位的权值
        delta_u2_temp(11,1) = a1*delta_u2_temp(11,1) + eta1 * delta_outputlayer * 1;
        %%更新权值
        for j=1:num_of_hidden_neurons
            u2(j,1) = u2(j,1) + delta_u2_temp(j,1);
            v2(j,1) = v2(j,1) + delta_v2_temp(j,1);
        end
        u2(11,1) = u2(11,1) + delta_u2_temp(11,1);

        %%back propagation 2: hidden layer
        for j=1:10 
            input_v = hidden_input(j,1);
            output_x = logsig(input_v);
            delta_hiddenlayer = exp(-1*input_v)*output_x^2 * delta_outputlayer * (2*u2(j,1)*output_x+v2(j,1));
            for k=1:2
                delta_u1_temp(j,k) = a1*delta_u1_temp(j,k) + eta1 * delta_hiddenlayer * data_train(i,k)^2;
                delta_v1_temp(j,k) = a2*delta_v1_temp(j,k) + eta2 * delta_hiddenlayer * data_train(i,k);
            end
            delta_u1_temp(j,3) = a1*delta_u1_temp(j,3) + eta1 * delta_hiddenlayer * 1;
            for k=1:2
                u1(j,k) = u1(j,k) + delta_u1_temp(j,k);
                v1(j,k) = v1(j,k) + delta_v1_temp(j,k);
            end
            u1(j,3) = u1(j,3) + delta_u1_temp(j,3);
        end  
    end
    disp(['online training程序总运行时间：',num2str(etime(clock,t1))]);
    
    %%--------------------STEP3 TESTING-----------------------------------
    error = 0;
    data_test_online=data_test;
    for i=1:size(data_test,1)
        raw_input1 = data_test(i,1);
        raw_input2 = data_test(i,2);
        raw_output = data_test(i,3);
            %%顺序执行
        %%存储hidden神经元的实际输入(sigmoid函数的自变量)
        hidden_input = rand(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_input(j,1) = input_of_hidden(raw_input1,raw_input2,u1(j,1),u1(j,2),u1(j,3),v1(j,1),v1(j,2));
        end 
        %%存储hidden神经元的输出，为中间十个神经元的输入的sigmoid函数输出
        hidden_output = rand(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_output(j,1) = logsig(hidden_input(j,1));
        end

        %%存储output神经元的输入
        output_input = input_of_output(u2,v2,hidden_output);
        %%存储output神经元的输出 完成feed forward，接下来开始back propagation
        output_output = logsig(output_input);
        data_test_online(i,3) = output_output;
        %%back propagation 1: output layer
        %%误差
        err = raw_output - output_output;
        error = error+err;
    end
    error = error/size(data_test,1);
    BP_online = error;
end
