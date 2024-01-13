% 
% detector = yolov4ObjectDetector("csp-darknet53-coco");
% disp(detector)  
% analyzeNetwork(detector.Network)

basenet = resnet50;
disp(basenet)
deepNetworkDesigner(basenet)
analyzeNetwork(basenet)