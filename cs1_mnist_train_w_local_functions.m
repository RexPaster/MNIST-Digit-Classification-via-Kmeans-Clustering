%% 
%% Case Study 1 Training Code by Rex Paster & Connor Strong
%This Code Uses a K-Means Algorithm to Create Clusters of Different Handwritten Numbers from a Training Data Set
% _Note: All custom functions are defined at the bottom of this file_ 

%% Clear Prior Workspace Data
clear all;
close all;
rng('shuffle'); %Shuffle The Random Number Seed For Better Randomness Between Runs

%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);
train = Normalize_Brightness(train); 

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200_woutliers.csv');
correctlabels = test(:,785); % store the correct test labels
test=test(:,1:784);
test(:,785)=zeros(200,1);


%% Set Parameters
k = 40; %Number of Clusters
max_iter= 100; %Max numbers of Iterations for the K-Means Algorithm
kmeans_reps = 5; %Repititions of the K-Means algorithm (Repitition with Lowest End Cost is used)

%% Call K-Means
[idx, centroids, cost] = kmeans_custom(train, k, max_iter, kmeans_reps); 

%% Plot K-Means Cost vs. Iterations
figure(Name="K-Means Cost vs. Iterations");
plot(cost, '-o');
xlabel("Iterations", "FontSize", 14)
ylabel("Cost","FontSize", 14)
title("K-Means Cost vs. Iterations", 'FontSize', 16, 'FontWeight','bold')

set(gcf, 'WindowState', 'maximized') %Fullscreen Figure For Better Figure Capturing with Publish Command


%% Label Centroids
for i = 1:length(centroids(:,1))
    centroid_labels(i) = mode(trainsetlabels(idx==i)); %Finds the Most Common Number of all Vectors Sorted into Cluster i
end


%% Plot Centroids
figure(Name="Centroids");
set(gcf, 'WindowState', 'maximized') %Fullscreen Figure For Better Figure Capturing with Publish Command
colormap('gray');
sgtitle("Centroids", 'FontSize', 16, 'FontWeight','bold')

plotsize = ceil(sqrt(length(centroids(:,1))));

for ind=1:length(centroids(:,1))
    
    centr=centroids(ind, 1:784);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    title("C" + ind + " | Represents " + string(centroid_labels(ind)));
end

%% Local Functions
%
% **Normalize_Brightness**
% Normalizes each vector's brightness so that its maximum value becomes 1.
%
% **kmeans_custom**
% Wrapper for the runkmeans function. Manages repetitions and picks the best run.
%
% **runkmeans**
% Standard K-Means algorithm: assigns vectors to centroids, computes costs,
% and updates centroids until convergence or max iterations.
%
% **initialize_centroids** (k-means++ initialization)
% Picks initial centroids using the k-means++ algorithm:
%
% Reference: Arthur & Vassilvitskii (2007),
% "k-means++: The Advantages of Careful Seeding"
% http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
%
% **assign_vector_to_centroid**
% Uses vectorized broadcasting to assign each vector to the nearest centroid
% and compute squared distances.
%
% **update_Centroids**
% Updates centroid positions by averaging vectors currently assigned
% to each cluster.
%
% _Special thanks to Ilan Kilman for suggesting the use of k-means++._

function normalized_data = Normalize_Brightness(data)
    normalized_data = zeros(size(data)); 

    % Loop over each vector (row) in the data
    for vector = 1:size(data, 1)
        % Find the maximum brightness for the current vector
        max_brightness = max(data(vector, :));

        % Normalize the vector by its maximum value
        normalized_data(vector, :) = data(vector, :) ./ max_brightness;
    end
end

function [idx, C, cost_iteration] = kmeans_custom(data, k, max_iter, reps)
    
   % Single run if reps not provided or <= 1
    if nargin <= 3 || reps <= 1
        [idx, C, cost_iteration] = runkmeans(data, k, max_iter);
        return;
    end

    % Multiple repetitions
    [idx, C, cost_iteration] = runkmeans(data, k, max_iter);
    for i = 1:reps
        [idx_temp, C_temp, cost_iteration_temp] = runkmeans(data, k, max_iter);

        % Keep the run with lower final cost
        if cost_iteration_temp(end) < cost_iteration(end)
            idx = idx_temp;
            C = C_temp;
            cost_iteration = cost_iteration_temp;
        end
    end
end

function [idx, centroids, cost_iteration] = runkmeans(data, k, max_iter)
    % Initialize centroids
    centroids = initialize_centroids(data, k);

    % Store K-Means cost at each iteration
    cost_iteration = [];

    % Run K-Means iterations
    for iter = 1:max_iter
        [idx, dists] = assign_vector_to_centroid(data, centroids);

        % Calculate cost for current iteration
        cost_iteration(iter) = sum(dists);

        % Update centroids based on current cluster assignments
        centroids = update_Centroids(data, k, idx);

        % Stop early if cost no longer decreases
        if iter > 1 && cost_iteration(iter) - cost_iteration(iter - 1) == 0
            break
        end
    end
end

function centroids = initialize_centroids(data, num_centroids)
%{
random_index=randperm(size(data,1));

centroids=data(random_index(1:num_centroids),:);
%}
% Pick the first centroid randomly
centroids = data(randi(length(data(:, 1)), 1), :);

    % Loop to pick the remaining centroids
    while length(centroids(:, 1)) < num_centroids
        distances_squared = zeros(length(data(:, 1)), 1);

        % Calculate squared distance of each vector to its nearest centroid
        for v = 1:length(data(:, 1))
            vector = data(v, :);
        
            % Set initial minimum distance to the first centroid
            min_distance = norm(vector - centroids(1, :));

            % Compare vector against all existing centroids to find the closest
            for i = 2:length(centroids(:, 1))
                min_distance = min(norm(vector - centroids(i, :)), min_distance);
            end      

            % Store the squared distance to the nearest centroid
            distances_squared(v) = min_distance^2;
        end

        % Choose the next centroid using a probability weighted by squared distance
        total = sum(distances_squared);
        threshold = total * rand();
        cumulative = 0;
        for i = 1:length(data(:, 1))
            cumulative = cumulative + distances_squared(i);
            if cumulative >= threshold
                centroids = [centroids; data(i, :)];
                break
            end
        end    
    end
end

function [index, vec_distance] = assign_vector_to_centroid(data, centroids)

    datalength = length(data(:, 1)); % Save the length of data for later

    % Permute the data matrix so that its rows extend in the z-dimension
    data = permute(data, [1, 3, 2]);

    % Repeat the data matrix along the y-dimension to match the number of clusters
    data = repmat(data, [1, length(centroids(:, 1)), 1]);

    % Permute the centroids matrix so that each centroid is in a different
    % column, and each element of the centroid is in the z-dimension
    centroids = permute(centroids, [3, 1, 2]);

    % Repeat the centroids matrix along the x-dimension to match the number of data vectors
    centroids = repmat(centroids, [datalength, 1, 1]);

    % Subtract the two matrices and take the squared norm along the z-dimension
    sqnorm_matrix = sum((data - centroids).^2, 3);

    % Find the index of the closest centroid for each vector by finding the
    % column with the smallest norm value for each row
    [~, index] = min(sqnorm_matrix, [], 2);

    % Create a vector vec_distance containing the norm of each vector to its nearest centroid
    vec_distance = zeros(length(data(:, 1)), 1); % Initialize vec_distance
    for row = 1:length(data(:, 1))
        vec_distance(row, 1) = sqnorm_matrix(row, index(row));
    end
end

function new_centroids = update_Centroids(data, k, idx)
    % Iterate through each centroid
    for centroid_index = 1:k
        % Iterate through each dimension of the centroid
        for centroid_vector_index = 1:size(data, 2)
            % Compute the mean of all data elements in the centroid_vector_index dimension 
            % that are assigned to the current centroid
            new_centroids(centroid_index, centroid_vector_index) = mean(data(idx == centroid_index, centroid_vector_index));
        end
    end
end

%% Final Notes on The Use of AI
%
% Artificial intelligence was used throughout the development of this code
% in the following ways:
% 
% - Error Analysis: troubleshooting code and pinpointing error sources
% - Optimization: suggesting ways to improve code speed and memory usage
% - Comment Grammar Checking: spelling and grammar checks on comments
% - Final Formatting Suggestions: function labeling, section header
%   formatting tips, etc.
%
% Please note that all code, with the exception of minimal single-line
% statements, was written by Rex Paster or Connor Strong, and is not a
% regurgitation of AI.

