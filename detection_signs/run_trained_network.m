close all; clear; clc;

load("detector.mat")

I = imread('real_test\2.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
 
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)