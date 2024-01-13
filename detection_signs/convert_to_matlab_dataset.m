function convert_to_matlab_dataset(input_path, output_file_name)
%#codegen

% T=readtable('detection/train_frames.csv');
T=readtable(input_path);

rows = height(T);

fileNames = {};
boundingBoxes= {};
for row = 1:rows 
    imageFilename = T{row, 8};
    % top-left point of the bounding box
    box_x1 = T{row, 3};
    box_y1 = T{row, 4};
    % bottom-right point of the bounding box
    box_x2 = T{row, 5};
    box_y2 = T{row, 6};
    % Calculating bounding box width and height
    box_width = box_x2 - box_x1;
    box_height = box_y2 - box_y1;
    fileNames{end+1} = imageFilename{1};
    boundingBoxes{end+1} = [box_x1, box_y1, box_width, box_height];
end   
convertedDataSet = table(fileNames(:),boundingBoxes(:), 'VariableName', {'imageFilename','object'});
% save('convertedDataSet.mat', 'convertedDataSet');
save(output_file_name, 'convertedDataSet');