function res = solve_constr_lsq(lambda,y,t)
% Solves the optimization problem for the coefficients Gamma_j of the Prony
% series
%
% Input: lambda = exponents of Prony series
%        y,t = data values and time grid of ACF data
%
% Output: res = tensor of coefficients Gamma_j

    dim = size(y,1);
    m = length(lambda);
    I = eye(dim*dim);
    
    % needed to ignore redundant rows in B later
    indmat = reshape(1:dim*dim,dim,dim);
    permut = indmat';
    P = I(permut(:),:);
    upperT = triu(indmat,1);
    evenInd = upperT(upperT~=0);

    y0 = reshape(y(:,:,1),[],1);

    % construct eq. matrices
    M0 = kron(ones(1,m),I);
    M1 = kron(lambda.',I);
    M2 = 1i*kron(lambda.^2.',I-P);
    M3 = kron(lambda.^3.',I+P);

    % lsq and constraint parameters
    B = [M0; M1; M2(evenInd,:);];
    d = [y0; zeros(dim*dim+length(evenInd),1)];
    G = kron( exp(lambda(:).*t ).',I );
    
    % solution of lsq
    a = constr_lsq(G,y(:),B,d);
    res = reshape(a,dim,dim,[]);

    % check for positive semidefiniteness of leading coefficient
    D = eig(reshape(M3*a,dim,dim),'vec');
    D(abs(D)<1e-10) = 0; % catch small eigenvalues below zero
    if min(real(D)) < 0
        error('Semidefiniteness constraint not satisfied');
    end
end

function sol = constr_lsq(A,b,B,d)
% solves the constrained least-squares problem
% 
% minimize ||Ax - b||_2     subject to Bx = d

   % check for unique solvability
   S = svd([A;B]);
   if sum(S(S<1e-14)) ~= 0
       warning('Constrained LSQ has no unique minimizer');
   end
   
   p = size(B,1);
   n = size(B,2);
   [QB,RB] = qr(B');
   Q1 = QB(:,1:p);
   Q2 = QB(:,p+1:end);
   x1 = Q1*(RB(1:p,1:p)'\d);
   [QA,RA] = qr(A*Q2);
   c = QA'*(b-A*x1);
   y2 = RA(1:n-p,1:n-p)\c(1:n-p);
   
   sol = x1 + Q2*y2;
end