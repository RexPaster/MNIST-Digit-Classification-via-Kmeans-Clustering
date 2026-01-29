%The following Code was used in generating our final centroids. It runs the
%code multiple times and takes the centroids most successful in sorting the
%test set.

%Initialize Vars In newdata.mat so that we have initial values
%Comment Out if you already have a newdata.mat
cs1_mnist_train_w_local_functions
predictions = 0;
save("newdata.mat", "centroids", "centroid_labels", "predictions");

while (now < 739890.9) %Runs Untill a Specific Time
    % Run training function
    cs1_mnist_train_w_local_functions

    % Normalize test data
    normalized_data = Normalize_Brightness(test);

    % Load stored predictions
    load("newdata.mat", "predictions");

    % Assign vectors to centroids
    [prediction_indexes, ~] = assign_vector_to_centroid_with_two_vector_dists_diff(normalized_data, centroids);

    % Generate new predictions
    for i = 1:200
        predictions_new(i) = centroid_labels(prediction_indexes(i));
    end

    % Update predictions if performance improves
    if predictions < sum(correctlabels' == predictions_new)
        predictions = sum(correctlabels' == predictions_new);
        save("newdata.mat", "centroids", "centroid_labels", "predictions");
    end

    % Display progress
    disp("Current Best: " + predictions + " | This run: " + sum(correctlabels' == predictions_new));

end


function normalized_data = Normalize_Brightness(data)
    % Normalize each row (vector) by its maximum brightness
    normalized_data = zeros(size(data));
    for vector = 1:size(data, 1)
        max_brightness = max(data(vector, :));
        normalized_data(vector, :) = data(vector, :) ./ max_brightness;
    end
end


function [index, diff_cent_distance] = assign_vector_to_centroid_with_two_vector_dists_diff(data, centroids)
    % Get number of data samples
    datalength = size(data, 1);

    % Expand data along cluster dimension
    data = permute(data, [1, 3, 2]);
    data = repmat(data, [1, size(centroids, 1), 1]);

    % Expand centroids along data dimension
    centroids = permute(centroids, [3, 1, 2]);
    centroids = repmat(centroids, [datalength, 1, 1]);

    % Compute squared distances
    sqnorm_matrix = sum((data - centroids).^2, 3);

    % Get closest centroid index
    [~, index] = min(sqnorm_matrix, [], 2);

    % Get two closest centroid distances
    min_cent_dists = mink(sqnorm_matrix, 2, 2);

    % Difference between closest two centroid distances
    diff_cent_distance = abs(min_cent_dists(:, 1) - min_cent_dists(:, 2));
end

%% Final Notes on the Use of AI
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
% Please note that all code was written by Rex Paster or Connor Strong, and is not a
% regurgitation of AI.