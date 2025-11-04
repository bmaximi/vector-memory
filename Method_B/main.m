clear, close all

outfile = fopen('diagnostic.out','w');

%%%%%% 1. Setting up the data
data_xx = importdata('vxvxc.txt');
data_xy = importdata('vxvyc.txt');
data_yx = importdata('vyvxc.txt');
data_yy = importdata('vyvyc.txt');

phi(1,1,:) = data_xx(:,2); phi(1,2,:) = data_xy(:,2);
phi(2,1,:) = data_yx(:,2); phi(2,2,:) = data_yy(:,2);
tt = data_xx(:,1).';

%%%%%% 2. Sampling the VACF data
n2 = 30;
dim = 2;
L = chol(phi(:,:,1));
for i=2:length(tt)
    phi(:,:,i) = inv(L)'*phi(:,:,i)*inv(L);
end
phi(:,:,1) = eye(dim);

twidth = 50;
t = tt(1:twidth:twidth*n2);
y = phi(:,:,1:twidth:twidth*n2);
tau = t(2);

fprintf(outfile, 'Number of data points: N = %1i\n',n2);
fprintf(outfile, 'Time step width: tau = %.1e\n',tau);

%%%%%% 3. Evaluate generating function on equicirceular grid with radius r and
% number of grid points g >> N
r = (1e6)^(1/n2);
g = 100;
[omega, F] = genFunc(y,r,g);

%%%%%% 4. AAA approximation

tol = 1e-4;
pol = blockaaa(omega,F,tol);
lambda = calc_exponents(pol,tau);

%%%%%% 5. Optimization of Gamma_j

res = solve_constr_lsq(lambda,y,t);

% diagnostic output
fprintf(outfile,'%1i negative poles duplicated\n\n',sum(length(lambda)-length(pol)));

lambdaOut = zeros(2*length(lambda),1);
lambdaOut(1:2:end) = real(lambda);
lambdaOut(2:2:end) = imag(lambda);
fprintf(outfile,'Exponents of Prony Series:\n');
fprintf(outfile,'  %7.4f %+7.4fi\n',lambdaOut);
fprintf(outfile,'\n');

%%%%%%% 6. Compute Markov parameters A,G and Sigma
A = constr_sysmat(lambda,res);
m = size(A,1);
fprintf(outfile,'Size of auxiliary subspace: %1i', m-dim);
fprintf(outfile,'\n\n');

[Sigma, G, outfile] = solve_Lure(A,dim,outfile);

disp('Markov parameters of system successfully reconstructed.');

%%%%%%% 7. Approximation anaylsis

app = 0*phi;
for i=1:length(tt)
    app(:,:,i) = eye(dim,m)*expm(tt(i)*A)*eye(m,dim);
end

% error interpolation points
tmp = app(:,:,1:twidth:twidth*n2) - phi(:,:,1:twidth:twidth*n2);
for i=1:n2
    errvec(i) = norm(tmp(:,:,i),'fro');
end
err = max(errvec);

% error interpolation interval
tmp = app(:,:,1:twidth*n2) - phi(:,:,1:twidth*n2);
for i=1:n2
    errvec(i) = norm(tmp(:,:,i),'fro');
end
err2 = max(errvec);

% error total time interval
tmp = app - phi;
for i=1:length(tt)
    errvec(i) = norm(tmp(:,:,i),'fro');
end
err3 = max(errvec);
