function X = basis_trafo(U,V)
% Computes a basis transformation X such that
% XV = ES^(1/2) and X'*E = US^(-1/2) with S = U*V

    d = size(V,2);
    S = U'*V;
    [Q,~,~] = svd(V);
    X = [S^(-1/2)*U'; Q(:,d+1:end)'];
end