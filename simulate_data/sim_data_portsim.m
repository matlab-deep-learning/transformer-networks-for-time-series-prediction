% Specify to only read open and close data
opts = detectImportOptions("data/SONY.csv");
opts.SelectedVariableNames = [2 5];

% Import data for Sony, Pepsi, and Merck
aapl = readmatrix("data/SONY.csv", opts);
pep = readmatrix("data/PEP.csv", opts);
mrk = readmatrix("data/MRK.csv", opts);

% Calculate average of open and close for each day
aapl = mean(aapl, 2);
mrk = mean(mrk, 2);
pep = mean(pep, 2);

% Import dates for data
[~, text, ~] = xlsread('data/SONY.csv');
dates = datetime(char(text(2:end, 1)), 'InputFormat', 'MM/dd/yyyy');
DateRange = dates <= datetime(2019,01,01);
dates = dates(DateRange);

% Clean data
data = [aapl mrk pep];
if sum(ismissing(data))
    data = rmmissing(data);
end

startPrice = 100;
numObs = 504;
numSim = 1;
retIntervals = 1;
% numAssets = 5;
numAssets = 3;

returns = data(end-numObs:end, :) - data(end-numObs, :);
expReturn = mean(returns) / 100;
expCov = cov(returns) / 100;

% expReturn     = [0.0246  0.0189  0.0273  0.0141  0.0311]/100;
% sigmas        = [0.9509  1.4259  1.5227  1.1062  1.0877]/100;
% correlations  = [1.0000  0.4403  0.4735  0.4334  0.6855
%                  0.4403  1.0000  0.7597  0.7809  0.4343
%                  0.4735  0.7597  1.0000  0.6978  0.4926
%                  0.4334  0.7809  0.6978  1.0000  0.4289
%                  0.6855  0.4343  0.4926  0.4289  1.0000];
% expCov = corr2cov(sigmas, correlations);

rng('default')
retExpected = portsim(expReturn, expCov, numObs, ...
    retIntervals, numSim, 'Expected');

weights = ones(numAssets, 1)/numAssets;
portRetExpected = zeros(numObs, numSim);
for i = 1:numSim
    portRetExpected(:, i) = retExpected(:, :, i)*weights;
end

portExpected = ret2tick(portRetExpected, repmat(startPrice, 1, numSim));
plot(portExpected)