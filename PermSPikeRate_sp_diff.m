session_id = 3; % session ID

method = 'abs'; %ratediff...clustering %abs...perm 
root = 'C:\Users\pine5\OneDrive - The University of Tokyo\code\Data';
cell_type = 'D2';       
mouse = '307-2_re';
fp = get_file_path(root, cell_type, mouse, session_id);

output = load(fp.sp);
output = output.S;
spiketime = output > 0;
spiketime = movsum(spiketime,10,1);
zero_cols = all(spiketime==0, 1);
bh{1} = readtable(fp.bh1);
bh{2} = readtable(fp.bh2);
bh{3} = readtable(fp.bh3);

% check size
n(1) = size(bh{1},1);
n(2) = size(bh{2},1);
n(3) = size(bh{3},1);

assert(size(spiketime,1)==sum(n));
N = [0,cumsum(n)];
n_cell = size(spiketime,2);
frameN = size(spiketime,1);
cluster = zeros(frameN, 1);

% split into 3 parts
for i = 1:2
    sp{i} = spiketime(N(i)+1:N(i+1),:);
end

for bi = 1:2 %2 % loop across 5 min blocks
    A1_5 = bh{bi}.A1_5;
    N1_5 = bh{bi}.N1_5;
    tempcluster =  cluster(N(bi)+1:N(bi+1),:);
    tempcluster(A1_5 == 1) = 1;
    tempcluster(N1_5 == 1) = 2;
    cluster(N(bi)+1:N(bi+1),:) = tempcluster;
end

global root_dir
declare_root_dir;
dir1 = sprintf('%s-%s',cell_type, mouse);
saveFolder = fullfile(root_dir,'PermResults',dir1);
filename = sprintf('cluster_session%d.mat',session_id);

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

savepath = fullfile(saveFolder, filename);
save(savepath, 'cluster');

%%
frameN = n(1)+n(2); %n(1)+n(2)
data_mat = (spiketime(1:frameN,:))';
nNeuron = size(data_mat,1);
nPerms = 1e3;
nSamples = size(data_mat,2);

%% for A N only
cluN = 2;

%% Precompute indices for each cluster
cluster_indices = cell(cluN, 1);
for clu = 1:cluN
    cluster_indices{clu} = find(cluster == clu);
end

%% Calculate cluster_event using precomputed indices
cluster_event = zeros(nNeuron, cluN);
for clu = 1:cluN
    % refer to the precomputed indices
    ind = cluster_indices{clu};
    for n = 1:nNeuron
        cluster_event(n, clu) = sum(data_mat(n, ind));
    end
    rate_cluster_event(:, clu) = cluster_event(:, clu)./length(ind);
end

alpha = 0.05;
pval_upper = zeros(nNeuron,cluN);
pval_lower = zeros(nNeuron,cluN);
sig = zeros(nNeuron,3); %cluN
%shuffleのところ、neuronごとにやった方がいい
% rng default
rng('default');
shiftAmounts = randperm(nSamples,nPerms);

cluster_event_shuffled = zeros(nPerms,cluN);

switch method
    case 'abs'
        for n = 1:nNeuron

            if 1%mod(n, 100)==0
                fprintf('%d/%d\n',n,nNeuron);
            end

            for m = 1:nPerms
                data_shuffled_temp = circshift(data_mat(n,:),shiftAmounts(m));
                for clu = 1:cluN
                    ind = cluster_indices{clu};
                    cluster_event_shuffled(m, clu) = sum(data_shuffled_temp(ind));
                end
            end

            for k = 1:cluN
                pval_upper(n,k) = mean(cluster_event_shuffled(:,k)>cluster_event(n,k));
                pval_lower(n,k) = mean(cluster_event_shuffled(:,k)<cluster_event(n,k));
            end
        end

    case 'rateabs'
        for n = 1:nNeuron

            if 1%mod(n, 100)==0
                fprintf('%d/%d\n',n,nNeuron);
            end


            for m = 1:nPerms
                data_shuffled_temp = circshift(data_mat(n,:),shiftAmounts(m));
                for clu = 1:cluN
                    ind = cluster_indices{clu};
                    rate_cluster_event_shuffled(m, clu) = sum(data_shuffled_temp(ind))./length(ind);
                end
            end

            for k = 1:cluN
                pval_upper(n,k) = mean(rate_cluster_event_shuffled(:,k)>rate_cluster_event(n,k));
                pval_lower(n,k) = mean(rate_cluster_event_shuffled(:,k)<rate_cluster_event(n,k));
            end
        end

    case 'diff'
        diff_cluster_event = cluster_event(:,1) - cluster_event(:,2);
        for n = 1:nNeuron

            if 1%mod(n, 100)==0
                fprintf('%d/%d\n',n,nNeuron);
            end

            for m = 1:nPerms
                data_shuffled_temp = circshift(data_mat(n,:),shiftAmounts(m));
                for clu = 1:cluN
                    ind = cluster_indices{clu};
                    cluster_event_shuffled(m, clu) = sum(data_shuffled_temp(ind));
                end
            end
            diff_cluster_event_shuffled = cluster_event_shuffled(:,1) - cluster_event_shuffled(:,2);
            for k = 1
                pval_upper(n,k) = mean(diff_cluster_event_shuffled(:,k)>diff_cluster_event(n,k));
                pval_lower(n,k) = mean(diff_cluster_event_shuffled(:,k)<diff_cluster_event(n,k));
            end
        end

    case 'ratediff'
        diff_cluster_event = rate_cluster_event(:,1) - rate_cluster_event(:,2);
        sig(:,3) = diff_cluster_event;
        for n = 1:nNeuron

            if 1%mod(n, 100)==0
                fprintf('%d/%d\n',n,nNeuron);
            end

            for m = 1:nPerms
                data_shuffled_temp = circshift(data_mat(n,:),shiftAmounts(m));
                for clu = 1:cluN
                    ind = cluster_indices{clu};
                    cluster_event_shuffled(m, clu) = sum(data_shuffled_temp(ind));
                    rate_cluster_event_shuffled(:, clu) = cluster_event_shuffled(:, clu)./length(ind);
                end
            end
            diff_cluster_event_shuffled = rate_cluster_event_shuffled(:,1) - rate_cluster_event_shuffled(:,2);
            for k = 1
                pval_upper(n,k) = mean(diff_cluster_event_shuffled(:,k)>diff_cluster_event(n,k));
                pval_lower(n,k) = mean(diff_cluster_event_shuffled(:,k)<diff_cluster_event(n,k));
            end
        end
end

%% Holm-Bonferroni correction

use_ind = [1 2]; % A N only
clu_used = length(use_ind);
switch method
    case {'abs','rateabs'}
        pval_use = pval_upper(:,use_ind);
        res = zeros(nNeuron,clu_used);
        for n = 1:nNeuron
            [pval_sorted, pval_order] = sort(pval_use(n,:)); 
            if zero_cols(n)
                continue
            end
            for k = 1:clu_used
                if pval_sorted(k)>= alpha/clu_used
                    break
                else
                    res(n,pval_order(k)) = 1;
                end
            end
        end

        for k = 1:clu_used
            sig(:,use_ind(k)) = res(:,k);
        end

        for clu = 1:cluN
            if sum(cluster==clu) == 0
                sig(:,clu) = 0;
            end
        end
    case {'diff','ratediff'}
        pval_use = [pval_upper(:,1), pval_lower(:,1)];
        p_2tail = min(1,2*min(pval_use(:,1),pval_use(:,2)));

        for n = 1:nNeuron
            if p_2tail(n,1) <= alpha
                if pval_use(n,1) < pval_use(n,2)
                    sig(n,1) = 1;
                else
                    sig(n,2) = 1;
                end
            end
        end

end

for clu = 1:cluN
    if sum(cluster==clu) == 0
        sig(:,clu) = 0;
    end
end

%% 
Component = sig;
filename1 = sprintf('Component_2_session%d_rdiff2tail.mat',session_id);   %Component_2session%d_rdiff2tail.mat
filename2 = sprintf('pval_upper_2_session%d_rdiff2tail.mat',session_id);
savepath1 = fullfile(saveFolder, filename1);
savepath2 = fullfile(saveFolder,filename2);
save(savepath1, 'Component');
save(savepath2, 'pval_upper');
%end
