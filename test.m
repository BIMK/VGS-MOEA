clc;
clear;
close all;
addpath(genpath(cd));


dataNames = {'SRBCT'};

k = 10;

for dataName = dataNames
	disp(dataName);

	load(dataName{1});
	data = full(data);
	% normalize
	data = (data - min(data, [], 1)) ./ (max(data, [], 1) - min(data, [], 1));
	data(isnan(data)) = 0;

	[m, featNum] = size(data);

	% crossvalid
	indices = crossvalind('Kfold', m, k);

	for i = 1 : 30
		disp(['K Fold ', int2str(i)]);
% 		crossvalid
		testIdx = indices == i;
		trainIdx = ~testIdx;
		testData = data(testIdx, :);
		testLabel = label(testIdx, :);
		trainData = data(trainIdx, :);
		trainLabel = label(trainIdx, :);

		tic;
		t1 = clock;
		[x, errTr, selFeatNum] = VGEA(trainData, trainLabel, dataName{1}, i);
		t2 = clock;
		toc;
		time = etime(t2, t1);
		
		accTr = 1 - errTr;

		save(strcat('result-', dataName{1}, '-', num2str(i)), 'x', 'errTr', 'selFeatNum', 'accTr', 'time');
	end
end
