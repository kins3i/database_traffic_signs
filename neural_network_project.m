imds = imageDatastore("images", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% TODO:
% 1) zmienić rozmiar na 32x32
% 2) augmentacja danych według kodu w kaggle (bez obracania!)





layers = [
    imageInputLayer([32 32 3],"Name","imageinput")
    convolution2dLayer([5 5],60,"Name","conv_1","Padding","same")
    convolution2dLayer([5 5],60,"Name","conv_2","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same")
    convolution2dLayer([3 3],30,"Name","conv_3","Padding","same")
    convolution2dLayer([3 3],30,"Name","conv_4","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same")
    flattenLayer("Name","flatten")
    reluLayer("Name","relu")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];