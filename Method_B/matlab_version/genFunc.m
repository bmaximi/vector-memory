function [omega, F] = genFunc(y,r,g)
    % Evaluates the generating function on an equiangular grid with radius
    % r and number of grid points g for data values y.
    
    d = size(y,1); N = size(y,3);
    
    omega = r*exp(2*pi*1i*(0:g-1)/g);
    F = zeros(d,d,g);
    for i=1:g
        for l=1:N
            F(:,:,i) = F(:,:,i) + y(:,:,l)*omega(i)^(-l);
        end
    end
end