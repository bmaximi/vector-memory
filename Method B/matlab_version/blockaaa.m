function poles = blockaaa(Z,F,tol)
% matrix-valued aaa rational approximation of data F on grid Z
%
% Input: F = tensor of data values
%        Z = vector of equiangular grid points
%        tol = error tolerance for aaa iteration
%
% Output: poles = vector of poles of the aaa approximation

    % initialize aux. variables
    Z = Z(:);
    M = length(Z);
    J = 1:M;

    % select the real data points for first approximation
    z = [Z(1); Z(M/2+1)];
    f(:,:,1) = F(:,:,1); f(:,:,2) = F(:,:,M/2+1);
    J(J==1) = []; J(J==M/2+1) = [];
    m = 2;

    L = constr_loewner(J,F,f,Z,z);
    [~,~,V] = svd(L,0);

    w1 = V(:,end);
    w = symm_weights(w1);

    R = compute_app(w,F,f,Z,z);
    errvec = 0*Z;
    for i=1:M
       errvec(i) = norm(F(:,:,i) - R(:,:,i),'fro');
    end
    [~, ind] = max(errvec);
    err = inf;

    while (m <= M/2 && err > tol)
       % choose next support point in greedy fashion + complex conjugate point
       m = m + 2;
       cc = M - ind + 2;
       z = [z; Z(ind); Z(cc)];
       f(:,:,m-1) = F(:,:,ind); f(:,:,m) = F(:,:,cc);
       J(J==ind) = []; J(J==cc) = [];

       % solve optimization problem for the weights
       L = constr_loewner(J,F,f,Z,z);
       [~,~,V] = svd(L,0);

       w1 = V(:,end);
       w = symm_weights(w1);

       % compute new rational approximation + approximation error
       R = compute_app(w,F,f,Z,z);
       errvec = 0*Z;
       for i=1:M
          errvec(i) = norm(F(:,:,i) - R(:,:,i),'fro');
       end
       [err, ind] = max(errvec);
    end

    poles = calc_poles(z,w);
    poles = poles(abs(poles)<1);
end

function L = constr_loewner(J,F,f,Z,z) 
    d = size(F,1);
    m = length(z);
    
    L = zeros(length(J)*d^2,m);
    for j=1:m
        tmp = zeros(length(J)*d,d);
        for k=1:length(J)
            tmp((k-1)*d+1:k*d,:) = ( F(:,:,J(k)) - f(:,:,j) ) / (Z(J(k)) - z(j));
        end
        L(:,j) = tmp(:);
    end
end

function W = symm_weights(W1)
    % Symmetrization of the weights to obtain a real-valued rational
    % function
    n = length(W1);
    W2 = W1;
    
    for i=3:2:n
        W2(i) = W1(i+1);
        W2(i+1) = W1(i);
    end
    
    if norm(W1 + W2) > 1e-12
        W = W1 + conj(W2);
        W = W / norm(W);
    else
        W = 1i*W1/norm(W1);
    end
end

function R = compute_app(w,F,f,Z,z)
    % Computes a rational approximation in barycentric form given the
    % interpolation set (z,w,f) for a grid Z
    R = F;
    
    N = 0*F; D = 0*Z;
    for k=1:length(Z)
        for l=1:length(z)
            N(:,:,k) = N(:,:,k) + (w(l)*f(:,:,l))/(Z(k) - z(l));
            D(k) = D(k) + w(l) / (Z(k) - z(l));
        end
    end
    for k=1:length(Z)
        if ismember(Z(k),z) == 0
            R(:,:,k) = N(:,:,k) / D(k);
        end
    end
end

function poles = calc_poles(z,w)
    % Computes the poles of a rational function in barycentric form
    m = length(z);
    B = eye(m+1); B(1,1) = 0;
    E = [0, w.'; ones(m,1), diag(z)];
    
    poles = eig(E,B);
end
