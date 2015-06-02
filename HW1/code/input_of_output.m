function out_input = input_of_output(u,v,input)
    temp = 0;    
    for i=1:10
        temp = temp + u(i)*input(i)^2 + v(i)*input(i);
    end
    temp = temp + 1*u(11);
    out_input = temp;
end