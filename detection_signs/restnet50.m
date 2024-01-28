inputSize = [224 224 3];


% convert_to_matlab_dataset('detection/train_frames_copy.csv','train_frames.mat');
% convert_to_matlab_dataset('detection/test_frames_copy.csv','test_frames.mat');

% save('own_train.mat', 'own_train')
train_data = importdata("own_train.mat");
% validation_data = importdata("test_frames.mat");

% Add the fullpath to the local vehicle data folder.
% train_data.imageFilename = fullfile(pwd, train_data.imageFilename);
% validation_data.imageFilename = fullfile(pwd,validation_data.imageFilename);



rng("default");
shuffledIndices = randperm(height(train_data));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = train_data(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = train_data(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = train_data(shuffledIndices(testIdx),:);


% Training dataset
imdsTrain = imageDatastore(trainingDataTbl{:,"imageFilename"});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,"object"));

% Validation dataset
imdsValidation = imageDatastore(validationDataTbl{:,"imageFilename"});
bldsValidation = boxLabelDatastore(validationDataTbl(:,"object"));

imdsTest = imageDatastore(testDataTbl{:,"imageFilename"});
bldsTest = boxLabelDatastore(testDataTbl(:,"object"));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

validateInputData(trainingData);
validateInputData(validationData);

% Display one image 
% display = true;
% if display 
%     data = read(validationData);
%     I = data{1};
%     bbox = data{2};
%     annotatedImage = insertShape(I,"Rectangle",bbox);
%     annotatedImage = imresize(annotatedImage,2);
%     figure
%     imshow(annotatedImage)
%     reset(validationData);
% end

% Resize DataSets
% trainingData = transform(trainingData,@(data)preprocessData(data,inputSize));
% validationData = transform(validationData,@(data)preprocessData(data,inputSize));

% Anchor boxes
rng("default")
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, inputSize));
numAnchors =7;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:7,:)
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
featureExtractionLayers = "activation_40_relu";
detector = yolov4ObjectDetector(dlnet,classes,anchorBoxes,DetectionNetworkSource=featureExtractionLayers);

% Do augumentation
% TODO: Augumented data is not added to the training set. We probably want to
% change that in the future
doAugmentation = true;

if doAugmentation   
    augmentedTrainingData = transform(trainingData,@augmentData);
    % augmentedData = cell(4,1);
    % for k = 1:4
    %     data = read(augmentedTrainingData);
    %     augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    %     reset(augmentedTrainingData);
    % end
    % figure
    % montage(augmentedData,BorderSize=10)
end

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));


% Training options
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=16,...
    L2Regularization=0.0005,...
    MaxEpochs=200,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=50,...
    ValidationData=preprocessedValidationData);

options2 = trainingOptions("sgdm", ...
    InitialLearnRate=1e-4, ...
    MiniBatchSize=16,...
    MaxEpochs=300, ...
    BatchNormalizationStatistics="moving",...
    ResetInputNormalization=false,...
    VerboseFrequency=20, ...
    ValidationFrequency=20,...
    ValidationData=preprocessedValidationData);

doTraining = true;
if doTraining       
    % Train the YOLO v4 detector.
    [detector,info] = trainYOLOv4ObjectDetector(preprocessedTrainingData,detector,options2);
else
    % Load pretrained detector for the example.
    detector = downloadPretrainedYOLOv4Detector();
end

% Simple test
I = imread("real_test/test3.png");
I = imresize(I,inputSize(1:2));
[bboxes,scores,labels] = detect(detector,I);

I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)
