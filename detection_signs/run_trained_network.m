clear; clc;

I = imread('real_test\test9.png');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
 
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)