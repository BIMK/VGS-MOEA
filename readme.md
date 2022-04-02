# VGS-MOEA

the code of "A Variable Granularity Search Based Multi-Objective Feature Selection Algorithm for High-Dimensional Data Classification (IEEE Transactions on Evolutionary Computation) doi: https://doi.org/10.1109/TEVC.2022.3160458" 

- VGEA.m: the code of algorithm.

- test.m: you an use test.m to start the algorithm.

- CalcObjs.m: the code to calculate objective values (error reatio and feature numbers).

the output of the algorithm is 

``` matlab
[x, errTr, selFeatNum] = VGEA(trainData, trainLabel, dataName{1}, i);
```

x is the individuals,

errTr is the error ratio,

selFeatNum is the number of selected features.
