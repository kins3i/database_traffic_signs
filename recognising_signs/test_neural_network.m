clear; clc;

load("net_augment.mat")
load("net_not_augm.mat")

our_sign = "A-7";

imds_Test = imageDatastore("znaki", ...
    'IncludeSubfolders',true, ...
    "FileExtensions",".png", ...
    "LabelSource","foldernames", ...
    'ReadFcn', @my_readDatastoreImage);

imds_Test = shuffle(imds_Test);

labels = imds_Test.Labels;


Y_notaugm = classify(net2, imds_Test);
% y_pred_notaugm = predict(net2, imds_Test);

for i = 1:length(Y_notaugm)
    if Y_notaugm(i) == our_sign
        classified_out_notaugm(i) = "yes";
    else
        classified_out_notaugm(i) = "no";
    end
end



Y_augmented = classify(net, imds_Test);
% y_pred_augment = predict(net, imds_Test);

for i = 1:length(Y_augmented)
    if Y_augmented(i) == our_sign
        classified_out_augm(i) = "yes";
    else
        classified_out_augm(i) = "no";
    end
end

classified_out_augm = categorical(classified_out_augm);
classified_out_notaugm = categorical(classified_out_notaugm);

figure()
cm_notaugm = confusionchart(imds_Test.Labels, classified_out_notaugm);

figure()
cm_augment = confusionchart(imds_Test.Labels, classified_out_augm);
