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
data = read(validationData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Create a YOLO v4 Object Detector Network

inputSize = [608 608 3];
className = "object";

reset(validationData);

% Resize DataSets
% trainingData = transform(trainingData,@(xx)preprocessData(xx,inputSize));
% validationData = transform(validationData,@(yy)preprocessData(yy,inputSize));

% Anchor boxes
rng("default")
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, inputSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };

% Created detector
detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,InputSize=inputSize);
augmentedTrainingData = transform(trainingData,@augmentData);

% Show augmentedData
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)
% Training options

options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=4,...
    L2Regularization=0.0005,...
    MaxEpochs=5,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=1000,...
    CheckpointPath=tempdir,...
    ValidationData=validationData);


doTraining = true;
if doTraining       
    % Train the YOLO v4 detector.
    [detector,info] = trainYOLOv4ObjectDetector(trainingDataForEstimation,detector,options);
else
    % Load pretrained detector for the example.
    detector = downloadPretrainedYOLOv4Detector();
end

I = imread("test/test2.png");
[bboxes,scores,labels] = detect(detector,I);

I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)