
%PATH_TO_EXECUTABLES are logreg execs as on website
path1 = getenv('PATH');
path1 = [path1 PATH_TO_EXECUTABLES];
setenv('PATH', path1);

%PATH_TO_DATA is data file 
fid = fopen(PATH_TO_DATA);
rawData = textscan(fid, '%f %f %d');
X = [rawData{1}, rawData{2}];
y = rawData{3};
mmwrite('ex_X',X); %mmwrite() mmread() files have been provided
mmwrite('ex_b',y); %creates ex_X, ex_b files in model_iono
system('l1_logreg_train -s ex_X ex_b 0.01 model_iono'); %creates model_iono file in pwd

model_iono = mmread('model_iono');
system('l1_logreg_classify -t ex_b model_iono ex_X result_iono');
result_iono = mmread('result_iono');