function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
m = size(X, 1);
centroids = zeros(K, size(X, 2));

% get the randomly-selected indices
indices = randperm(m);
indices = indices(1:K);
% choose the subset of a you want
centroids = X(indices, :);
% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%








% =============================================================

end

