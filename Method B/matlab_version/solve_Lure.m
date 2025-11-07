function [Sigma, G, outfile] = solve_Lure(A,d,outfile)
% Solves the equation A*Sigma+Sigma*A' = -G*G' for G and Sigma.
%
% Input: A = system matrix
%        d = dimension of the system
%        outfile = file for diagnostic output
%
% Output: Sigma, G = covariance matrix and direction of Brownian motion
%         outfile = file for diagnostic output

    % partition A
    B = A(1:d,d+1:end)';
    C = -A(d+1:end,1:d);
    A0 = A(d+1:end,d+1:end);

    N = size(A0,1);

    % perform reduction step
    X = basis_trafo(B,C);
    XAX = X*A0/X;
    D1 = -XAX(1:d,1:d);
    B1 = XAX(1:d,d+1:end)';
    C1 = -XAX(d+1:end,1:d);
    A1 = XAX(d+1:end,d+1:end);

    R1 = D1 + D1';

    % Check if reduced system is solvable
    if min(eig(R1))<0
        error('*** Lure equations are not solvable: D>0');
    end

    if (size(A1,1) > 0) & (max(real(eig(A1))) >= 0)
        error('*** A0 is not stable! Lure equations are not solvable!')
    end

    % Solve (reduced) regular Lur'e system
    P = A1 - C1*(R1\B1');
    Q1 = B1*(R1\B1');
    Q2 = C1*(R1\C1');

    Sigma1 = Riccati(P,Q1,Q2);

    % check for numerical errors
    err = norm(P*Sigma1 + Sigma1*P' + Sigma1*B1/R1*B1'*Sigma1 + C1/R1*C1')/...
            norm(P*Sigma1);

    if err > 1e-6
        error('*** Riccati equation not satisfied');
    end
    if (size(Sigma1,1) > 0) & (norm(Sigma1-Sigma1') > 1e-5)
        error('*** Riccati solution is not symmetric')
    end

    % Transform reduced system back to full system
    K1 = chol(R1)';
    L1 = (C1 - Sigma1*B1)/K1';

    XL = [K1; L1]; L = X\XL;
    XSigma0X = [eye(d), zeros(d,N-d); zeros(N-d,d), Sigma1];
    Sigma0 = X\XSigma0X/X';
    Sigma0 = (Sigma0 + Sigma0')/2; % nullify numerical rounding errors

    Sigma = [eye(d), zeros(d,N); zeros(N,d), Sigma0];
    G = [zeros(d); L];

    fprintf(outfile,'Error of Lyapunov equation (non-transformed system): %8.6e\n',norm(A*Sigma+Sigma*A'+G*G')/norm(A*Sigma));
end