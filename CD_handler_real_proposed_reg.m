clear all;
addpath('./facebook100_cooked/');
% Caltech36_A.mat, Reed98_A.mat, Haverford76_A.mat, Simmons81_A.mat, Swarthmore42_A.mat, Amherst41_A.mat, 
% Bowdoin47_A.mat, Hamilton46_A.mat, USFCA72_A.mat, Trinity100_A.mat, Oberlin44_A.mat, Williams40_A.mat, 
% Wellesley22_A.mat, Smith60_A.mat, Middlebury45_A.mat, Vassar85_A.mat, Mich67_A.mat, Pepperdine86_A.mat, 
% Colgate88_A.mat, Wesleyan43_A.mat, Santa74_A.mat, Brandeis99_A.mat, Bucknell39_A.mat, Rice31_A.mat, Howard90_A.mat
rng(1, 'twister');
dataset = 'email';
model = 'MVCD';
fprintf('dataset is %s.\n', dataset);
number_hops = 6;
X = load(strcat(dataset, '_A.mat'));
X = double(X.A);
Adjmat = X;
tempX = speye(size(X));
M = zeros(size(X, 1), size(X, 2), number_hops);
%% =================construct data tensor=======================
for k = 1:number_hops
    tempX = tempX * X; % $X^k$.
    M(:, :, k) = NormAdjac(tempX);
end

X = tensor(M);
%% =================construct data tensor=======================

target = load(strcat(dataset, '_target.mat'));
target = target.T;
n = size(X, 1);
m = size(X, 3);
% heads = 2; % search # head in grid
k = length(unique(target)); % set as same as the ground-truth (as required by reviewer #3)
fprintf('number of nodes is %d, number of community is %d.\n', n, k);
paralist = [1e-3 1e-2 1e-1 1 10];
headlist = [1 2 3];
dup = linspace(1, 10, 10);
[para1, para2, para3, para4, tau] = ndgrid(paralist, paralist, paralist, headlist, dup);
nmi_max = zeros(numel(tau), 1);
purity_max = zeros(numel(tau), 1);
pre_max = zeros(numel(tau), 1);
rec_max = zeros(numel(tau), 1);
f1_max = zeros(numel(tau), 1);
comSubsets = cell(numel(tau), 1);

parfor ind1 = 1:numel(tau)
    rng(42 * ind1);
    eta = para1(ind1);
    lamd = para2(ind1);
    gamma = para3(ind1);
    heads = para4(ind1);
    fprintf('This is %d-th search. (total %d.)\n', ind1, numel(tau));
    [U, Z, V, B] = MVCD_reg(Adjmat, X, n, m, k, heads, lamd, eta, gamma, false);
    [~, q] = max(U, [], 2);
    comSubsets{ind1, 1} = q;
    purity = cluster_Purity2(target, q); % 1 or 2? 2
    [p, r, f1] = cluster_F1(target, q);
    nmi = cluster_NMI(target, q);

    purity_max(ind1) = purity;
    pre_max(ind1) = p;
    rec_max(ind1) = r;
    f1_max(ind1) = f1;
    nmi_max(ind1) = nmi;
    fprintf('Purity is %f, F1 is %f, NMI is %f.\n', purity, f1, nmi);
end

clc;
reshape_purity = mean(reshape(purity_max, [], 10), 2);
reshape_f1 = mean(reshape(f1_max, [], 10), 2);
reshape_nmi = mean(reshape(nmi_max, [], 10), 2);
disp(max(reshape_purity));
disp(max(reshape_f1));
disp(max(reshape_nmi));
% fprintf('%s, %f, %f, %f, %f.\n', dataset, max(purity_max), max(pre_max), max(rec_max), max(f1_max));
% save(strcat('./ANTD_regresless/', model, '_', dataset, '_purity.mat'), 'purity_max');
% % save(strcat('./ANTD_regresless/', model, '_', dataset, '_pre.mat'), 'pre_max');
% % save(strcat('./ANTD_regresless/', model, '_', dataset, '_rec.mat'), 'rec_max');
% save(strcat('./ANTD_regresless/', model, '_', dataset, '_f1.mat'), 'f1_max');
% save(strcat('./ANTD_regresless/', model, '_', dataset, '_nmi.mat'), 'nmi_max');
% save(strcat('./ANTD_regresless/', model, '_', dataset, '_comSubsets.mat'), 'comSubsets');
