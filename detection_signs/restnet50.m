inputSize = [224 224 3];

convert_to_matlab_dataset('detection/train_frames_copy.csv','train_frames.mat');
convert_to_matlab_dataset('detection/test_frames_copy.csv','test_frames.mat');

train_data = importdata("train_frames.mat");
validation_data = importdata("test_frames.mat");

% Add the fullpath to the local vehicle data folder.
train_data.imageFilename = fullfile(pwd, train_data.imageFilename);
validation_data.imageFilename = fullfile(pwd,validation_data.imageFilename);

% Training dataset
imdsTrain = imageDatastore(train_data{:,"imageFilename"});
bldsTrain = boxLabelDatastore(train_data(:,"object"));

% Validation dataset
imdsValidation = imageDatastore(validation_data{:,"imageFilename"});
bldsValidation = boxLabelDatastore(validation_data(:,"object"));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);

validateInputData(trainingData);
validateInputData(validationData);

% Display one image 
display = true;
if display 
    data = read(validationData);
    I = data{1};
    bbox = data{2};
    annotatedImage = insertShape(I,"Rectangle",bbox);
    annotatedImage = imresize(annotatedImage,2);
    figure
    imshow(annotatedImage)
    reset(validationData);
end

% Resize DataSets
% trainingData = transform(trainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

% Anchor boxes
rng("default")
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, inputSize));
numAnchors = 6;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

% Created detector with trannsfer learning
classes = {'object'};

basenet = resnet50;
imageSize = inputSize;
layerName = basenet.Layers(1).Name;
newinputLayer = imageInputLayer(imageSize,'Normalization','none','Name',layerName);
lgraph = layerGraph(basenet);
lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');
lgraph = replaceLayer(lgraph,layerName,newinputLayer);
dlnet = dlnetwork(lgraph);
featureExtractionLayers = ["activation_22_relu","activation_40_relu"];
detector = yolov4ObjectDetector(dlnet,classes,anchorBoxes,DetectionNetworkSource=featureExtractionLayers);

% Do augumentation
% TODO: Augumented data is not added to the training set. We probably want to
% change that in the future
doAugmentation = false;

if doAugmentation   
    augmentedTrainingData = transform(trainingData,@augmentData);
    augmentedData = cell(4,1);
    for k = 1:4
        data = read(augmentedTrainingData);
        augmentedData{k} = insertShape(data{1},"rectangle",data{2});
        reset(augmentedTrainingData);
    end
    figure
    montage(augmentedData,BorderSize=10)
end

% Training options
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=4,...
    L2Regularization=0.0005,...
    MaxEpochs=50,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=1000,...
    ValidationData=validationData);

options2 = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=16,...
    MaxEpochs=40, ...
    BatchNormalizationStatistics="moving",...
    ResetInputNormalization=false,...
    VerboseFrequency=30);

doTraining = true;
if doTraining       
    % Train the YOLO v4 detector.
    [detector,info] = trainYOLOv4ObjectDetector(trainingDataForEstimation,detector,options);
else
    % Load pretrained detector for the example.
    detector = downloadPretrainedYOLOv4Detector();
end

% Simple test
I = imread("test/test2.png");
[bboxes,scores,labels] = detect(detector,I);

I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)
