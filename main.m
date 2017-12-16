%% MAIN PROGRAM
%==========================================================================
% Juan M. Gandarias, Jesús M. Gómez-de-Gabriel and Alfonso J. García-Cerezo
% Robotics and Mechatronics Research Group
% System Engineering and Automation Department
% University of Málaga, Spain
% ------------------------------------------------------------------------- 
% The input dataset is in the 'Images' folder. Then, the DCNN-SVM approach
% using the AlexNet network is used to classify the data.
% The Neural Network Toolbox and the Alexnet Toolbox are needed
% =========================================================================

% Clear the workspace
clear

%% Load a  Pre-trained CNN
% AlexNet has been trained on the ImageNet dataset previously, which has 
% 1000 object categories and 1.2 million training images
convnet = alexnet;

%% Auxiliar Variables Initialization
rate = zeros(3,20);     % Matrix of recognition rates
best_mat = zeros(10);   % Best matrix of each 20 samples
maximum = 0;                % maximumimum recognition rate achieved
n_config = 3;           % Number of possibles configurations
n_samples = 20;         % Number of samples of each experiment


%% 3 Experiments Loop
for i=1:n_config*n_samples
	%Reset the count each 20 samples
    if i==1 || i==n_samples+1 || i==2*n_samples+1
        n=1;    
    end
        
    %Download the Data Set
    if i<=n_samples
        %% Rigid configuration
        setDir = fullfile('Dataset/Images/rigid');
        j = 1;
        % Store the dataset in a variable
        imds = imageDatastore(setDir,'IncludeSubfolders',true,...
            'LabelSource','foldernames','FileExtensions','.jpg');

        % Summarize the number of images per category.
        tbl = countEachLabel(imds);
        minSetCount = min(tbl{:,2});

    elseif i>n_samples && i<=2*n_samples
        %% Semi-rigid configuration
        setDir = fullfile('Dataset/Images/semi-rigid');
        j = 2;
        % Store the dataset in a variable
        imds = imageDatastore(setDir,'IncludeSubfolders',true,...
            'LabelSource','foldernames','FileExtensions','.jpg');

        % Summarize the number of images per category.
        tbl = countEachLabel(imds);
        minSetCount = min(tbl{:,2});

        % Randomize the data
        imds = splitEachLabel(imds, minSetCount, 'randomize');
    else
        %% Flexible configuration
        setDir = fullfile('Dataset/Images/flexible');
        j = 3;
        % Store the dataset in a variable
        imds = imageDatastore(setDir,'IncludeSubfolders',true,...
            'LabelSource','foldernames','FileExtensions','.jpg');

        % Summarize the number of images per category.
        tbl = countEachLabel(imds);
        minSetCount = min(tbl{:,2});

        % Randomize the data
        imds = splitEachLabel(imds, minSetCount, 'randomize');
    end
       
    % Randomize the data
    imds = splitEachLabel(imds, minSetCount, 'randomize');

    %% Pre-process Images For CNN
    % AlexNet can only process RGB images that are 227-by-227.

    % Set the ImageDatastore ReadFcn
    imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
    
    % Split the data into the training and the test sets
    [trainingSet, validationSet] = splitEachLabel(imds, 0.4, 'randomize');

    %% Train A Multiclass SVM Classifier Using CNN Features
    % Next, use the CNN image features to train a multiclass SVM classifier. A
    % fast Stochastic Gradient Descent solver is used for training by setting
    % the |fitcecoc| function's 'Learners' parameter to 'Linear'.

    % Extract features from one of the deeper layers using the
    % |activations| method. Selecting the layer right before the
    % classification layer ('fc7'). 
    featureLayer = 'fc7';
    trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
        'MiniBatchSize', 32, 'OutputAs', 'columns');

    % Get training labels from the trainingSet
    trainingLabels = trainingSet.Labels;

    % Train multiclass SVM classifier using a fast linear solver, and set
    % 'ObservationsIn' to 'columns' to match the arrangement used for training
    % features.
    classifier = fitcecoc(trainingFeatures, trainingLabels, ...
        'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn',...
        'columns');

    %% Evaluate Classifier
    % Repeat the procedure used earlier to extract image features from
    % |validationSet|. The test features can then be passed to the 
    % classifier to measure the accuracy of the trained classifier.

    % Extract test features using the CNN
    testFeatures = activations(convnet, validationSet, featureLayer,...
        'MiniBatchSize',32);

    % Pass CNN image features to trained the classifier
    predictedLabels = predict(classifier, testFeatures);

    % Get the known labels
    testLabels = validationSet.Labels;

    % Tabulate the results using a confusion matrix.
    confMat = confusionmat(testLabels, predictedLabels);

    % Convert confusion matrix into percentage form
    confMatrix = bsxfun(@rdivide,confMat,sum(confMat,2));

    % Calculate the recognition rate
    recognition_rate = mean(diag(confMatrix));
    rate(j,n) = recognition_rate;
    % Get the best recognition rate and ir respective confusion matrix
    if recognition_rate > maximum
        maximum = recognition_rate;
        best_mat = confMatrix;
    end
    
    % Plot the best confusion matrix of each configuration
    if i==n_samples
        best_rigid_mat = best_mat;
        PlotConfMatrix(best_rigid_mat)
        title('Rigid Configuration')
        maximum = 0;
    elseif i==(n_config-1)*n_samples 
        best_semirigid_mat = best_mat;
        PlotConfMatrix(best_semirigid_mat)
        title('Semi-Rigid Configuration')
        maximum = 0;
    elseif i==n_config*n_samples
        best_flexible_mat = best_mat;
        PlotConfMatrix(best_flexible_mat)
        title('Flexible Configuration')
    end
    
    n=n+1;
end

%% Plot the box
experiment = ["Rigid","Semi-Rigid","Flexible"];
figure
boxplot(rate', experiment)
title('Classification Experiment')
ylabel('Recognition Rate [%]')

%% Calculate the mean recognition rate of each configuration
mean_rigid = mean(rate(1,:));
mean_semirigid = mean(rate(2,:));
mean_flex = mean(rate(3,:));

%% Calculate the best recognition rate of each configuration
best_rigid = max(rate(1,:));
best_semirigid = max(rate(2,:));
best_flex = max(rate(3,:));

%% Save the workspace
save('experiment')
