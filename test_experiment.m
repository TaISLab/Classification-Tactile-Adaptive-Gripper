%% TEST THE RESULTS OF THE EXPERIMENT
%==========================================================================
% Juan M. Gandarias, Jesús M. Gómez-de-Gabriel and Alfonso J. García-Cerezo
% Robotics and Mechatronics Research Group
% System Engineering and Automation Department
% University of Málaga, Spain
% ------------------------------------------------------------------------- 
% This script load the workspace of the experiment to reproduce and plot 
% the output data without having to train the method from scratch.
% =========================================================================

% Clear the current workspace
clear

% Load the experiment workspace
load('experiment.mat')

% Plot the best confusion matrix of each configuration
PlotConfMatrix(best_rigid_mat)
title('Rigid Configuration')
PlotConfMatrix(best_semirigid_mat)
title('Semi-Rigid Configuration')
PlotConfMatrix(best_flexible_mat)
title('Flexible Configuration')

% Plot the box
experiment = ["Rigid","Semi-Rigid","Flexible"];
figure
boxplot(rate', experiment)
title('Classification Experiment')
ylabel('Recognition Rate [%]')

% Calculate the mean recognition rate of each configuration
mean_rigid = mean(rate(1,:));
mean_semirigid = mean(rate(2,:));
mean_flex = mean(rate(3,:));

% Calculate the best recognition rate of each configuration
best_rigid = max(rate(1,:));
best_semirigid = max(rate(2,:));
best_flex = max(rate(3,:));