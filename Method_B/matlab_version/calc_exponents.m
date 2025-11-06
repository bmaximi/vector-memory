function lambda = calc_exponents(pol,tau)
% Computes the exponents from a given set of poles
%
% Input: pol = poles of rational approximation
%        tau = time width
%
% Output: lambda = exponents of Prony series

    % find positive and negative real exponents
    pos_ind = real(pol)>0 & abs(imag(pol))<1e-8;
    neg_ind = real(pol)<0 & abs(imag(pol))<1e-8;
    
    [ccPol, ~] = sort(log(pol(~pos_ind&~neg_ind)));
    
    % transform negative real poles to complex conjugate pairs
    negPol = [real(log(pol(neg_ind))) + 1i*pi; real(log(pol(neg_ind)))-1i*pi;];
    [sorted_negPol, ~] = sort(negPol);
    
    logpol = [log(pol(pos_ind)); ccPol; sorted_negPol];
    lambda = logpol/tau;
end