clear; 
clc;
imds_Train = imageDatastore("images/train", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames', ...
    'ReadFcn', @my_readDatastoreImage);

imds_Val = imageDatastore("images/test", ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames', ...
    'ReadFcn', @my_readDatastoreImage);

img_imds1 = read(imds_Val);
whos img_imds1

% TODO:
% 2) augmentacja danych według kodu w kaggle (bez obracania!)



imdsTrain = shuffle(imds_Train);
imdsValidation = shuffle(imds_Val);

targetSize = [32 32];

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
    'RandScale',[0.2 1], ...
    'RandXShear', [-10 0], ...
    'RandXTranslation', [0 0.2*targetSize(1)], ...
    'RandYTranslation', [0 0.2*targetSize(2)] );



auimds = augmentedImageDatastore(targetSize,imdsTrain,'DataAugmentation',augmenter);

val_augm = augmentedImageDatastore(targetSize,imdsValidation,'DataAugmentation',augmenter);



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
    fullyConnectedLayer(92,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];


options_augm = trainingOptions("adam", ...
    MaxEpochs=10, ...
    VerboseFrequency=130,...
    ValidationFrequency=130,...
    Plots="training-progress", ...
    OutputNetwork="best-validation-loss", ...
    ValidationData=val_augm, ...
    Verbose=true);

options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    VerboseFrequency=130,...
    ValidationFrequency=130,...
    Plots="training-progress", ...
    OutputNetwork="best-validation-loss", ...
    ValidationData=imdsValidation, ...
    Verbose=true);

%     ValidationPatience=5, ...


% ypred_im = predict(net, test_images);

net2 = trainNetwork(imdsTrain,layers,options);

YPred2 = classify(net2,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy2 = mean(YPred2 == YValidation)


net = trainNetwork(auimds,layers,options_augm);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)


save('net_not_augm.mat', 'net2')
save('net_augment.mat', 'net')
