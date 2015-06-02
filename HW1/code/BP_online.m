
%%�ĸ�����������a1,a2,eta1,eta2
function BP_online = BP_online(p1,p2,p3,p4)
    data_train = load('two_spiral_train.txt');
    data_test = load('two_spiral_test.txt');



    %%------------------------STEP1 INITIALIZE-----------------------------
    %%hidden����Ԫ�趨Ϊ10��
    num_of_hidden_neurons = 10; 
    %%ÿ�е�����Ԫ�������ǵ�һ�������u�͵ڶ��������u��ƫ��ֵ1��Ȩֵ��һ��10�б�ʾ��ʮ��hidden����Ԫ
    u1 = rand(num_of_hidden_neurons,3).*2-1;
    delta_u1_temp = zeros(num_of_hidden_neurons,3); %%�洢temp delta u
    %%ÿ�е�2��Ԫ�������ǵ�һ�������v�͵ڶ��������v��һ��10�б�ʾ��ʮ��hidden����Ԫ
    v1 = rand(num_of_hidden_neurons,2).*2-1;
    delta_v1_temp = zeros(num_of_hidden_neurons,2);

    %%11��Ȩֵ�ֱ���10��hidden����Ԫ��������uȨֵ�Լ�ƫ��ֵ1��Ȩֵ
    u2 = rand(num_of_hidden_neurons+1,1).*2-1;
    delta_u2_temp = zeros(num_of_hidden_neurons+1,1);
    %%10��Ȩֵ�ֱ���10��hidden����Ԫ��������vȨֵ
    v2 = rand(num_of_hidden_neurons,1).*2-1;
    delta_v2_temp = zeros(num_of_hidden_neurons,1);

    %%��ʼ������
    a1 = p1;
    a2 = p2;
    eta1 = p3;
    eta2 = p4;


    
    %%------------------------STEP2 TRAINING-------------------------------
    %%ѭ��ѵ��
    t1 = clock;
    for i=1:size(data_train,1)
        raw_input1 = data_train(i,1);
        raw_input2 = data_train(i,2);
        raw_output = data_train(i,3);

        %%˳��ִ��
        %%�洢hidden��Ԫ��ʵ������(sigmoid�������Ա���)
        hidden_input = zeros(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_input(j,1) = input_of_hidden(raw_input1,raw_input2,u1(j,1),u1(j,2),u1(j,3),v1(j,1),v1(j,2));
        end 
        %%�洢hidden��Ԫ�������Ϊ�м�ʮ����Ԫ�������sigmoid�������
        hidden_output = zeros(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_output(j,1) = logsig(hidden_input(j,1));
        end

        %%�洢output��Ԫ������
        output_input = input_of_output(u2,v2,hidden_output);
        %%�洢output��Ԫ����� ���feed forward����������ʼback propagation
        output_output = logsig(output_input);

        %%back propagation 1: output layer
        %%���
        err = raw_output - output_output;

        delta_outputlayer = err * exp(-1 * output_input) * output_output^2;
        for j=1:num_of_hidden_neurons
            delta_u2_temp(j,1) = a1*delta_u2_temp(j,1) + eta1 * delta_outputlayer * hidden_output(j,1)^2;
            delta_v2_temp(j,1) = a2*delta_v2_temp(j,1) + eta2 * delta_outputlayer * hidden_output(j,1);
        end
        %%ƫ��λ��Ȩֵ
        delta_u2_temp(11,1) = a1*delta_u2_temp(11,1) + eta1 * delta_outputlayer * 1;
        %%����Ȩֵ
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
    disp(['online training����������ʱ�䣺',num2str(etime(clock,t1))]);
    
    %%--------------------STEP3 TESTING-----------------------------------
    error = 0;
    data_test_online=data_test;
    for i=1:size(data_test,1)
        raw_input1 = data_test(i,1);
        raw_input2 = data_test(i,2);
        raw_output = data_test(i,3);
            %%˳��ִ��
        %%�洢hidden��Ԫ��ʵ������(sigmoid�������Ա���)
        hidden_input = rand(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_input(j,1) = input_of_hidden(raw_input1,raw_input2,u1(j,1),u1(j,2),u1(j,3),v1(j,1),v1(j,2));
        end 
        %%�洢hidden��Ԫ�������Ϊ�м�ʮ����Ԫ�������sigmoid�������
        hidden_output = rand(num_of_hidden_neurons,1);
        for j=1:num_of_hidden_neurons
            hidden_output(j,1) = logsig(hidden_input(j,1));
        end

        %%�洢output��Ԫ������
        output_input = input_of_output(u2,v2,hidden_output);
        %%�洢output��Ԫ����� ���feed forward����������ʼback propagation
        output_output = logsig(output_input);
        data_test_online(i,3) = output_output;
        %%back propagation 1: output layer
        %%���
        err = raw_output - output_output;
        error = error+err;
    end
    error = error/size(data_test,1);
    BP_online = error;
end
