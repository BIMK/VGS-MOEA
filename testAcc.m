function acc = testAcc(trainData, trainLabel, testData, testLabel, X)
	%% Test On KNN
	X = X > 0.6;
	if (sum(X) == 0)
		acc = 0;
		return;
	end

	knnK = 1;
	try
		mdl = ClassificationKNN.fit(trainData(:, X), trainLabel, 'NumNeighbors', knnK);
	catch err
		acc = 0;
		return;
	end
	y = predict(mdl, testData(:, X));
	y = y == testLabel;
	acc = sum(y) / numel(y);
end