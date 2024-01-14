clear; 
clc;
imds = imageDatastore("images", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

img_imds1 = read(imds);

whos img_imds1

% TODO:
% 1) zmienić rozmiar na 32x32
% 2) augmentacja danych według kodu w kaggle (bez obracania!)


% numWholeTrain = length(imds_resized.UnderlyingDatastores{1, 1}.Labels);
percTrainFiles = 0.7;
[imdsTrain,imdsValidation] = splitEachLabel(imds,percTrainFiles,'randomize');


% TODO 1:
targetSize = [32 32];

imdsTrain = transform(imdsTrain,@(x) imresize(x,targetSize));
imdsTrain = transform(imdsTrain,@(x) im2double(x));

imdsValidation = transform(imdsValidation,@(x) imresize(x,targetSize));
imdsValidation = transform(imdsValidation,@(x) im2double(x));


% imdsTrain = transform(imdsTrain, @commonPreprocessing);
% imdsValidation = transform(imdsValidation, @commonPreprocessing);


img_resized1 = read(imdsTrain);
whos img_resized1

% preview(imdsValidation)


% TODO 2:
% https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
% train_image_generator_aug = ImageDataGenerator(rescale=1./255,
%                                            shear_range=0.1,
%                                            zoom_range=0.2,
%                                            brightness_range=[0.5, 1.5],
%                                            width_shift_range=0.2,
%                                            height_shift_range=0.2,
%                                            channel_shift_range=0.2)
augmenter = imageDataAugmenter( ...
    'RandScale',[0.5 1], ...
    'RandXShear', [0, 0], ...
    'RandYShear', [0, 0]  );







layers = [
    imageInputLayer([32 32 3],"Name","imageinput")
    convolution2dLayer([5 5],60,"Name","conv_1","Padding","same")
    convolution2dLayer([5 5],60,"Name","conv_2","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same")
    convolution2dLayer([3 3],30,"Name","conv_3","Padding","same")
    convolution2dLayer([3 3],30,"Name","conv_4","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same")
%     flattenLayer("Name","flatten")
    reluLayer("Name","relu")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];


options = trainingOptions("adam", ...
    MaxEpochs=50, ...
    ValidationData=imdsValidation, ...
    ValidationPatience=5, ...
    Plots="training-progress", ...
    OutputNetwork="best-validation-loss", ...
    Verbose=false);


% net = trainNetwork(imdsTrain,layers,options);
% 
% YPred = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% accuracy = mean(YPred == YValidation)
















function dataOut = commonPreprocessing(data)

    dataOut = cell(size(data));
    for col = 1:size(data,2)
        for idx = 1:size(data,1)
            temp = single(data{idx,col});
            temp = imresize(temp,[32,32]);
            targetSize
            temp = rescale(temp);
            dataOut{idx,col} = temp;
        end
    end
end

