clc
clear
close all

% 文件夹路径
inputFolder = 'C:\Xiaohu Xu\LiveTissue\Image_Process\Images_Mat';
outputFolder = 'C:\Xiaohu Xu\LiveTissue\Image_Process\Images_Mat';

% 文件前缀
prefixes = {'Cali_1', 'Cali_2', 'Cali_3'};

% 遍历每个前缀
for p = 1:length(prefixes)
    prefix = prefixes{p};
    
    % 获取当前前缀的.mat文件列表
    matFiles = dir(fullfile(inputFolder, [prefix, '*.mat']));
    
    % 检查是否有文件
    if isempty(matFiles)
        fprintf('没有找到前缀为%s的文件\n', prefix);
        continue;
    end
    
    % 初始化存储数据的变量
    allData = [];
    
    % 遍历所有.mat文件并累加数据
    for i = 1:length(matFiles)
        matFilePath = fullfile(inputFolder, matFiles(i).name);
        data = load(matFilePath);
        
        % 假设.mat文件中的数据变量名为stack
        if isfield(data, 'stack')
            imageData = data.stack; % 26维度的图像数据
            
            % 检查数据是否是26维的
            if size(imageData, 3) == 26
                if isempty(allData)
                    allData = imageData;
                else
                    allData = allData + imageData; % 累加
                end
            else
                fprintf('文件 %s 不是26维的，跳过。\n', matFiles(i).name);
            end
        else
            fprintf('文件 %s 不包含变量 stack，跳过。\n', matFiles(i).name);
        end
    end
    
    % 计算平均值
    stack = allData / length(matFiles);
    
    % 保存平均值数据为.mat文件
    save(fullfile(outputFolder, [prefix, '.mat']), 'stack');
end

% 加载Water_1、Water_2和Water_3的.mat文件
Water_1 = load(fullfile(outputFolder, 'Cali_1.mat'));
Water_2 = load(fullfile(outputFolder, 'Cali_2.mat'));
Water_3 = load(fullfile(outputFolder, 'Cali_3.mat'));

% 确保数据都存在
if isfield(Water_1, 'stack') && isfield(Water_2, 'stack') && isfield(Water_3, 'stack')
    % 获取所有Water_1、Water_2、Water_3的26维数据
    data_1 = Water_1.stack;
    data_2 = Water_2.stack;
    data_3 = Water_3.stack;
    
    % 计算每一维度的中间值
    stack = median(cat(4, data_1, data_2, data_3), 4);
    
    % 保存最终的Water.mat文件
    save(fullfile(outputFolder, 'Cali.mat'), 'stack');
else
    fprintf('加载时出现问题。\n');
end
