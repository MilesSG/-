%%上海大学本科机械工程创新比赛答案（基于Matlab的深度学习分类器）
clc; clear; close all;
%% 准备训练数据
filepath = "E:\2020competition\Undergraduate Group Industrial Big Data Analysis Competition Questions\Train";
data = imageDatastore(filepath,'IncludeSubfolders',true,'LabelSource','foldernames');  % 导入数据，以文件夹名作为分类名
datagroup = data.Labels;  % 获取分类标签

[dataTrain,dataTest] = splitEachLabel(data,0.7);  % 划分训练集与测试集,训练集占train的70%，测试集占train的30%
audsTrain = augmentedImageDatastore([227 227],dataTrain);  % 修改图片大小以适应预训练网络
audsTest = augmentedImageDatastore([227 227],dataTest);
numClasses = numel(categories(data.Labels));  % 获取类别数

%% 创建迁移网络
net = alexnet;  % 导入预训练网络alexnet
layers = net.Layers;  % 获取层信息
fc = fullyConnectedLayer(numClasses);  % 修改全连接层，与输出类别数一致
layers(end - 2) = fc;
layers(end) = classificationLayer;

%% 设置训练选项
opts = trainingOptions('sgdm','InitialLearnRate',0.001);  % 优化器：sgdm, 学习率0.001

%% 执行训练
[pcbnet,info] = trainNetwork(audsTrain, layers, opts);


%% 使用经过训练的网络对测试图像进行分类
testPreds = classify(pcbnet,audsTest);

%% 评估性能
figure,
plot(info.TrainingLoss);  % 绘制损失函数变化曲线

testActual = dataTest.Labels;  % 获取训练集真实值
numCorrect = nnz(testPreds == testActual);  % 计算正确个数
fracCorrect = numCorrect/numel(testPreds);  % 计算准确率

figure,
confusionchart(dataTest.Labels,testPreds);  % 计算混淆矩阵
