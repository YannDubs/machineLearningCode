function [Iquant] = quantizeImage(I,b)
%
% QUANTIZEIMAGE compresses an image using only b bits for the color space
%   [Iquant] = quantizeImage(I,b)
%
[h,w,c]=size(I);
k=2^b;
% here reshape simply reshapes the matrix such that each pixel is a row
% and there are 3 columns containing the values of each color.
Ir = reshape(I,[h*w c]);
model = clusterKmeans(Ir,k,false);
y = model.predict(model,Ir);
IrNew=model.W(y,:);

% reshapes back to the initial image size
Iquant=reshape(IrNew,[h w c]);
end