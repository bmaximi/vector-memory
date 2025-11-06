function sysmat = constr_sysmat(exps, res)
% Computes the system matrix for a given Prony series

    % init parameters
    d = size(res,1);
    m = length(exps);
    r = 1;
    J = [];
    USigma = []; VSigma = [];

    % real exponents first
    while abs(imag(exps(r))) < 1e-8
        [U,S,V] = svd(res(:,:,r)); S = diag(S);
        for i=1:d
            if S(i) < 1e-12
                break
            end
            J(end+1,end+1) = real(exps(r));
            USigma(:,end+1) = sqrt(S(i))*real(U(:,i));
            VSigma(end+1,:) = sqrt(S(i))*real(V(:,i))';
        end
        r = r+1;
        if r > m
            break
        end
    end

    % complex conjugate pairs next
    while r < m
        [U,S,V] = svd(res(:,:,r)); S = diag(S);
        for i=1:d
            if S(i) < 1e-12
                break
            end
            J(end+1:end+2,end+1:end+2) = [real(exps(r)), imag(exps(r));...
                                         -imag(exps(r)), real(exps(r))];
            USigma(:,end+1) = sqrt(2*S(i))*real(U(:,i));
            USigma(:,end+1) = sqrt(2*S(i))*imag(U(:,i));
            VSigma(end+1,:) = sqrt(2*S(i))*real(V(:,i)');
            VSigma(end+1,:) = -sqrt(2*S(i))*imag(V(:,i)');
        end
        r = r+2;
    end
    
    X = basis_trafo(USigma',VSigma);
    sysmat = X*J*inv(X);

end