function SOM
    data = importdata('hw4-data.txt');
    min_data = min(data(:));
    max_data = max(data(:));
    centers = rand(5, 5, 2) * (max_data - min_data) - min_data;

    iters = [1 10 100 1000 2000];
    iter_max = 2000;
    alpha = 0.2;
    for i = 1 : iter_max
        for j = 1 : size(data, 1)
            x = data(j, :);
            x = reshape(x, [1 1 2]);
            x_data = repmat(x, [5 5 1]);
            dist = sum((centers - x_data).^2, 3);
            [~, index] = min(dist(:));

            mcol = ceil(index/5);
            mrow = mod(index, 5);
            if mrow == 0, mrow = 5; end

            [rs, cs] = ndgrid((1:5) - mrow, (1:5) - mcol);
            distance = rs.^2 + cs.^2;
            neighbor = (distance < 1.5);
            neighbor = repmat(neighbor, [1 1 2]);
            centers = centers + alpha * neighbor .* (x_data - centers);
        end
        alpha = alpha - 0.0001;
        if sum(i == iters)>0
            x = centers(:, :, 1);x = x(:);
            y = centers(:, :, 2);y = y(:);
            V = [x,y];
            E = zeros(25, 25);
            for i = 1:5:21
                for j = 1:4
                    E(i+j-1, i+j) = 1;
                end
            end
            for i = 1:5
                for j = 5:5:20
                    E(i+j-5, i+j)=1;
                end
            end
            gplot(E, V);
        end
    end
end
