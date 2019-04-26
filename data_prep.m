% ML LRC - Pilot

gauss_data = dlmread('twoGaussians.txt');

csvwrite('twoGaussians.csv', gauss_data);

% 11-Dimensional data
class_positive = gauss_data(gauss_data(:,end) == 1 ,1:end-1);
class_negative = gauss_data(gauss_data(:,end) ~= 1 ,1:end-1);

% data class label
label_positive = gauss_data(gauss_data(:,end) == 1 ,end);
label_negative = gauss_data(gauss_data(:,end) ~= 1 ,end);
