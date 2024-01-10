load transformer_demo_sim.mat

t = 0;
X = 100; % starting price
X = X(ones(nVariables, 1));

F = @(t, X) diag(expReturn) * X; % drift function
G = @(t, X) diag(X) * diag(sigma); % diffusion function

SDE = sde(F, G, 'Correlation', correlation, 'StartState', X); % stochastic differential equation to model market

nDays = length(dates)-1;
rng(123)
[S, T] = simByEuler(SDE, nDays, 'DeltaTime', 1);

varNames = ["Stock A" "Stock B" "Stock C"];
data = timetable(dates, ...
    S(:, 1), S(:, 2) ,S(:, 3), ...
    'VariableNames', varNames);

% Save data to .mat file
save("transformer_demo_sim", "data", "varNames", "-append")