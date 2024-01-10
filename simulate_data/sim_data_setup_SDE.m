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

returns = tick2ret(data);
nVariables = size(returns, 2);
expReturn = mean(returns);
sigma = std(returns);
correlation = corrcoef(returns);
nDays = length(data);

save("transformer_demo_sim", "nVariables", "expReturn", "sigma", "correlation", "dates")

t = 0;
% X = 100;
% X = X(ones(nVariables, 1));
X = [25; 50; 75];

F = @(t, X) diag(expReturn) * X;
G = @(t, X) diag(X) * diag(sigma);

SDE = sde(F, G, 'Correlation', correlation, 'StartState', X);

dt = 1;
rng(50)
[S, T] = simByEuler(SDE, nDays, 'DeltaTime', dt);
plot(T, S)
