clc
clear
close all

water = load([pwd,'\PBS.mat']);
water = water.stack;

cali = load([pwd,'\Cals.mat']);
cali = cali.stack;

folderPath = [pwd,'\Images'];
folderPathOutput = [pwd,'\Images_Flatfield'];

filePattern = fullfile(folderPath, 'Tissue*.mat');
matFiles = dir(filePattern);

% Smoothing the calibration image
windowSize = [25, 25]; 
sigma = 4;       
gaussianFilter = fspecial('gaussian', windowSize, sigma);

for i = 1:size(water, 3)
    caliFluid(:,:,i) = imfilter(cali(:,:,i)-water(:,:,i), gaussianFilter, 'symmetric');
    cali(:,:,i) = imfilter(cali(:,:,i), gaussianFilter, 'symmetric');
    water(:,:,i) = imfilter(water(:,:,i), gaussianFilter, 'symmetric');
end

for k = 1:length(matFiles)
    baseFileName = matFiles(k).name;
    fullFileName = fullfile(folderPath, baseFileName);

    matData = load(fullFileName);

    fieldName = fieldnames(matData);  
    multiSpectralImage = matData.(fieldName{1});  

    % Calibrates the images
    for i = 1:size(multiSpectralImage, 3)
        multiSpectralImage(:,:,i) = (multiSpectralImage(:,:,i)-water(:,:,i))./caliFluid(:,:,i);
    end

    % Save
    newFileName = fullfile(folderPathOutput, baseFileName);  
    matData.(fieldName{1}) = multiSpectralImage;  
    save(newFileName, '-struct', 'matData');  

end

disp('Done！');