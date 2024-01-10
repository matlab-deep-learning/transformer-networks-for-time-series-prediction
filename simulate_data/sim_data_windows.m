% Specify to only read open and close data
opts = detectImportOptions("data/JNJ.csv");
opts.SelectedVariableNames = [2 5];

% Import data for Sony, Pepsi, and Merck
jnj = readmatrix("data/JNJ.csv", opts);
pg = readmatrix("data/PG.csv", opts);
ko = readmatrix("data/KO.csv", opts);

% Calculate average of open and close for each day
jnj = mean(jnj, 2);
ko = mean(ko, 2);
pg = mean(pg, 2);

% Import dates for data
[~, text, ~] = xlsread('data/JNJ.csv');
dates = datetime(char(text(2:end, 1)), 'InputFormat', 'MM/dd/yyyy');
DateRange = dates <= datetime(2024,01,01);
dates = dates(DateRange);

% Clean data
data = [jnj ko pg];
if sum(ismissing(data))
    data = rmmissing(data);
end

returns = tick2ret(data);
[nDays, nVariables] = size(returns);
windowSize = 30;
nWindows = round(nDays / windowSize);
stockData = [];
X = [50; 55; 45];
t = 0;

for i=1:nWindows-1
    windowData = returns((1 + (i-1)*windowSize):windowSize*i, :);
    expReturn = mean(windowData);
    sigma = std(windowData);
    correlation = corrcoef(windowData);

    F = @(t, X) diag(expReturn) * X;
    G = @(t, X) diag(X) * diag(sigma);
    
    SDE = sde(F, G, 'Correlation', correlation, 'StartState', X);
    [S, ~] = simulate(SDE, windowSize-1, 'DeltaTime', 1);

    X = S(end, :)';
    stockData = [stockData; S];
end

varNames = ["Stock A" "Stock B" "Stock C"];
nSimDays = size(stockData, 1);
dates = dates(1:nSimDays);
data = timetable(dates, ...
    stockData(:, 1), stockData(:, 2), stockData(:, 3), ...
    'VariableNames', varNames);

plot(data, "dates", varNames)
legend(varNames, 'Location','Northwest')
title("Average Daily Price of Stocks")
ylabel("Price")
xlabel("Time")

save('transformer_demo', 'data', 'varNames')