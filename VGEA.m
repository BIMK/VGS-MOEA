function [pfs, errTr, selFeatNum] = VGEA(data, label, dataName, fold)
	%% Init
	featNum = size(data, 2);

	maxIt = 100;		% the max number of iterations
	maxIt2 = 10;		% the value of alpha
	popNum = 100;		% poplation size
	archiveNum = popNum;

	archive = [];
	archiveObjs = [];

	groupNum = round(10 * log2(featNum));
	firstStageEnd = maxIt2 * floor(log2(featNum / groupNum) + 1);
 
	% group
	[info, r] = SortFeature(data, label);
	idx = kmeans(info, groupNum);
	idx = idx';
	for i = 1 : groupNum
		g{i} = find(idx == i)';
	end
	% Init pop
	pop = rand(popNum, groupNum) > 0.5;
	p = r ./ featNum;

	% map group space to full features space
	x = group2x(g, pop, featNum);
	objs = CalcObjs(data, label, x);
	[archive, archiveObjs] = UpdateArchive(x, objs, archive, archiveObjs, archiveNum);

	%% Iteration
	it = 2;
	while it <= maxIt
		for it2 = 1 : maxIt2
			interPop = [pop; x2group(g, archive)];
			MatingPool = randi(size(interPop, 1), 1, popNum);
			pop2 = GeneticOperator(interPop(MatingPool, :), MatingPool > size(pop, 1), GetProb(g, p));

			x = group2x(g, pop2, featNum);
			objs2 = CalcObjs(data, label, x);
			[archive, archiveObjs] = UpdateArchive(x, objs2, archive, archiveObjs, archiveNum);
			[pop, objs] = EnvironmentalSelection([pop; pop2],  [objs; objs2], popNum); 

			disp(strcat("VGEA on ", dataName, " Fold ", num2str(fold), " Iter ", num2str(it)));
			it = it + 1;
			if it > maxIt
				break;
			end
		end

		if it > firstStageEnd
			x = group2x(g, pop, featNum);
			[archive, archiveObjs] = NSGAII(x, objs, popNum, it, maxIt, data, label, dataName, fold, ...
			archive, archiveObjs, archiveNum, r);
			it = inf;
		else
			% split
			[pop, objs, g] = Split(pop, objs, g, popNum, data, label);
		end
	end

	pf = archiveObjs;
	pfs = archive;
	errTr = pf(:, 1);
	selFeatNum = pf(:, 2);
end

function x2 = GetProb(g, x)
	popNum = size(x, 1);
	groupNum = numel(g);

	x2 = zeros(popNum, groupNum);
	for i = 1 : popNum
		for j = 1 : groupNum
			x2(i, j) = mean(x(i, g{j}));
		end
	end

end

function [archive, archiveObjs] = NSGAII(pop, objs, popNum, it, maxIt, data, label, dataName, fold,...
	archive, archiveObjs, archiveNum, r)

	while it <= maxIt
		interPop = [pop; archive];
		MatingPool = randi(size(interPop, 1), 1, popNum);
		pop2 = GeneticOperator2(interPop(MatingPool, :), MatingPool > size(pop, 1));

		objs2 = CalcObjs(data, label, pop2);
		[archive, archiveObjs] = UpdateArchive(pop2, objs2, archive, archiveObjs, archiveNum);
		[pop, objs] =  EnvironmentalSelection([pop; pop2],  [objs; objs2], popNum);

		disp(strcat("VGEA on ", dataName, " Fold ", num2str(fold), " Iter ", num2str(it)));
		it = it + 1;
	end
end

function pop2 = GeneticOperator(parent, isArchive, p)
	%% crossover and mutation
	[m, n] = size(parent);
	p1 = parent(1 : m / 2, :);
	p2 = parent(m / 2 + 1 : m, :);


	ia1 = isArchive(1 : m / 2);
	ia2 = isArchive(m / 2 + 1 : m);
	t = ia1 | ia2;
	pop2 = [
		GA1(p1(~t, :), p2(~t, :));
		GA2(p1(t, :), p2(t, :), ia1(t), ia2(t), p)
	];
end

function pop2 = GeneticOperator2(parent, isArchive)
	%% crossover and mutation
	[m, n] = size(parent);
	p1 = parent(1 : m / 2, :);
	p2 = parent(m / 2 + 1 : m, :);

	pop2 = GA1(p1, p2);
end

function Offspring = GA1(Parent1, Parent2)
	proC = 0.9;
	proM = 1;

	[N,D]   = size(Parent1);

	%% Genetic operators for binary encoding
	% One point crossover
	k = repmat(1:D,N,1) > repmat(randi(D,N,1),1,D);
	k(repmat(rand(N,1)>proC,1,D)) = false;
	Offspring1    = Parent1;
	Offspring2    = Parent2;
	Offspring1(k) = Parent2(k);
	Offspring2(k) = Parent1(k);
	Offspring     = [Offspring1;Offspring2];
	% Bitwise mutation
	Site = rand(2*N,D) < proM/D;
	Offspring(Site) = ~Offspring(Site);

end

function pop2 = GA2(p1, p2, ia1, ia2, p)
	proC = 0.9;
	proM = 1;

	[m, n] = size(p1);

	%% crossover
	np1 = p1 & p2;
	np2 = xor(p1, p2);
	np3 = zeros(m, n);
	% t = rand(m / 2, n) > rand();
	t = randi(n, m, n) > randi(n, m, 1);
	np3(t) = np2(t);
	np4 = xor(np2, np3);

	np3 = np1 | np3;
	np4 = np1 | np4;
	k = rand(m,1) > proC;
	np3(k, :) = p1(k, :);
	np4(k, :) = p2(k, :);
	pop2 = [np3; np4];

	%% mutation
	tt = zeros(size(p1));
	tt(ia1, :) = tt(ia1, :) | p1(ia1, :);
	tt(ia2, :) = tt(ia2, :) | p2(ia2, :);
	tt = [and(tt, xor(np1, np3)); and(tt,xor(np1, np4))];
	tt = 1 / n + tt .* p;
	% 	tt = 1 / n;
	t = rand(size(pop2)) < tt;
	pop2 = xor(pop2, t);

end

function [archive, archiveObjs] = UpdateArchive(pop, fit, archive, archiveObjs, archiveNum)

	[FrontNo, ~] = NDSort(fit,size(fit, 1));
	idx = min(fit(:, 1)) == fit(:, 1);
	idx = idx | FrontNo' == 1;
	archive = [archive; pop(idx, :)];
	archiveObjs = [archiveObjs; fit(idx, :)];
	
	[FrontNo, ~] = NDSort(archiveObjs,size(archiveObjs, 1));
	Front1 =  FrontNo' == 1;
	idx = min(archiveObjs(:, 1)) == archiveObjs(:, 1);
	idx = idx & ~Front1;
	
	Front1 = find(Front1);
	idx = find(idx);
	
	if numel(Front1) + numel(idx) > archiveNum
		idx = idx(randperm(numel(idx), archiveNum - numel(Front1)));
	end
	
	idx = union(Front1, idx);
	archive = archive(idx, :);
	archiveObjs = archiveObjs(idx, :);
	
% 	[~, uni] = unique(archive, 'rows');
%     archive = archive(uni, :);
%     archiveObjs = archiveObjs(uni, :);

	[archive, archiveObjs] = duplicate(archive, archiveObjs);
end

function [pop, fit] = duplicate(pop, fit)
	
	m = size(pop, 1);
	dup = zeros(m, 1);
	d = zeros(m, m);
	lens = sum(pop, 2);
	for i = 1 : m
		for j = i + 1 : m
			d(i, j) = sum(pop(i, :) + pop(j, :) == 2);
			if d(i, j) == min(lens(i), lens(j))
				if fit(i, 1) > fit(j, 1)
					dup(i) = 1;
				elseif fit(i, 1) < fit(j, 1)
					dup(j) = 1;
				else
					if lens(i) > lens(j)
						dup(i) = 1;
					else 
						dup(j) = 1;
					end
				end
			end
		end
	end
	
	pop = pop(dup == 0, :);
	fit = fit(dup == 0, :);

end


function x = group2x(g, pop, featNum)
	%% map group to full feature space
	x = zeros(size(pop, 1), featNum);
	for i = 1 : size(pop, 1)
		for j = 1 : size(g, 2)
			x(i, g{j}) = pop(i, j);
		end
	end
end

function x2 = x2group(g, x)
	popNum = size(x, 1);
	groupNum = numel(g);

	x2 = zeros(popNum, groupNum);
	for i = 1 : popNum
		for j = 1 : groupNum
			x2(i, j) = max(x(i, g{j}));
		end
	end

end

function [pop, fit, g] = Split(pop, fit, g, popNum, data, label)
	featNum = size(data, 2);
	x = group2x(g, pop, featNum);

	[m, n] = size(pop);
	num = 1;
	t3 = 1 : n;
	if t3
		for i = t3
			n1 = numel(g{i});
			if n1 < 2
				g2{num} = g{i};
				num = num + 1;
			else
				n2 = floor(n1 / 2);
				gt = g{i};
				g2{num} = gt(1 : n2);
				g2{num + 1} = gt(n2 + 1 : n1);
				num = num + 2;
			end
		end
	end
	g = g2;
	pop = x2group(g, x);
end

function [info, r] = SortFeature(data, label)
% Get infomation from dataset

    c = GetPearsonCorr(data, label);
    [~, r1] = sort(c, 'descend');
    [~, r1] = sort(r1, 'ascend');

    [su, ent] = GetSU(data, label);
    [~, r4] = sort(su, 'descend');
    [~, r4] = sort(r4, 'ascend');

    [~, r5] = sort(ent, 'descend');
    [~, r5] = sort(r5, 'ascend');

    % chi2 = getChi2(data, label);
    % [~, r6] = sort(chi2, 'descend');
    % [~, r6] = sort(r6, 'ascend');

    % r = r1 .* r4 .* r6;
    % r = r1 .* r4;
    r = (r1 + r4) ./ 2;
    % r = r4 .* r5;
    r = r';
    c = (c - min(c)) ./ (max(c) - min(c));
    su = (su - min(su)) ./ (max(su) - min(su));
    ent = (ent - min(ent)) ./ (max(ent) - min(ent));
    % chi2 = (chi2 - min(chi2)) ./ (max(chi2) - min(chi2));
    % info = [c, su, chi2];
    info = [c, su];

    info(isnan(info)) = 0;

end


function [su, ent] = GetSU(x, y)

su = zeros(size(x, 2), 1);
ent = zeros(size(x, 2), 1);
for i = 1 : size(x, 2)
    [su(i), ent(i)] = MItest(x(:, i), y);
end

end

function [mi, Ha] = MItest(a,b)
%culate MI of a and b in the region of the overlap part

%计算重叠部分
[Ma,Na] = size(a);
[Mb,Nb] = size(b);
M=min(Ma,Mb);
N=min(Na,Nb);

%初始化直方图数组
hab = zeros(256,256);
ha = zeros(1,256);
hb = zeros(1,256);

%归一化
if max(max(a))~=min(min(a))
    a = (a-min(min(a)))/(max(max(a))-min(min(a)));
else
    a = zeros(M,N);
end

if max(max(b))-min(min(b))
    b = (b-min(min(b)))/(max(max(b))-min(min(b)));
else
    b = zeros(M,N);
end

a = double(int16(a*255))+1;
b = double(int16(b*255))+1;

%统计直方图
for i=1:M
    for j=1:N
       indexx =  a(i,j);
       indexy = b(i,j) ;
       hab(indexx,indexy) = hab(indexx,indexy)+1;%联合直方图
       ha(indexx) = ha(indexx)+1;%a图直方图
       hb(indexy) = hb(indexy)+1;%b图直方图
   end
end

%计算联合信息熵
hsum = sum(sum(hab));
index = find(hab~=0);
p = hab/hsum;
Hab = sum(sum(-p(index).*log(p(index))));

%计算a图信息熵
hsum = sum(sum(ha));
index = find(ha~=0);
p = ha/hsum;
Ha = sum(sum(-p(index).*log(p(index))));

%计算b图信息熵
hsum = sum(sum(hb));
index = find(hb~=0);
p = hb/hsum;
Hb = sum(sum(-p(index).*log(p(index))));

%计算a和b的互信息
mi = Ha+Hb-Hab;

%计算a和b的归一化互信息
mi = hab/(Ha+Hb);
end

function v = getVar(data)

featNum = size(data, 2);

v = zeros(featNum, 1);

for i = 1 : featNum
    v(i) = var(data(:, i));
end

end


function chi2 = getChi2(data, label)

[m, n] = size(data);

chi2 = zeros(n, 1);
for i = 1 : n
    [~, chi2(i), ~, ~] = crosstab(data(:, i), label);
end

end


function c1 = GetPearsonCorr(data, label)

[m, n] = size(data);

c1 = zeros(n, 1);

for i = 1 : n
    c1(i) = abs(corr(data(:, i), label, 'type', 'Pearson'));
end

end