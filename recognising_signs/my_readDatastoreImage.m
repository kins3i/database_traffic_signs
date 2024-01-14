function data = my_readDatastoreImage(filename)
%READDATASTOREIMAGE Read file formats supported by IMREAD.
%
%   See also matlab.io.datastore.ImageDatastore, datastore,
%            mapreduce.

%   Copyright 2016-2020 The MathWorks, Inc.

    % Turn off warning backtrace before calling imread
    onState = warning('off', 'backtrace');
    c = onCleanup(@() warning(onState));
    N = 10;
    for iter = 1 : N
        try
            data = imread(filename);
            data = imresize(data, [32, 32]);
            data = im2double(data);
            break;
        catch ME
            pause(10^-6);
            if iter == N
                throw(ME);
            end
        end
    end
end
