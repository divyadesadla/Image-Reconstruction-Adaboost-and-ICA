clc;
clear;
clear all;

%% QUESTION 2.1


P = '/Users/divyadesadla/Desktop/CMU/CMUFall19/MLSP/Homework2/hw2_materials/hw2materials/problem2/Ifw1000';
D=dir(fullfile(P,'*.pgm'));
nimages=length(D);
for i=1:nimages
      image{i}=double(imread(fullfile(P,D(i).name)));      
end

[nrows,ncolumns] = size(image);
image = image(:);

X = [];
for j=1:length(image)
    X = [X image{j}(:)];  %% making matrix from image column vector
end

%% Calculating reconstruction error

[U,S,V] = svd(X);              %% to get eigenfaces from X
eigenface = U(:,1);
eigenface_image = reshape(U(:,1),64,64);
imagesc(eigenface_image);

csvwrite('eigenface.csv', eigenface);

reconstruction_error = [];
for k=1:100
    A = U(:,1:k)* S(1:k,:) * V';
    reconstruction_error =[reconstruction_error 1/1071 * norm((X - A),'fro')^2];
end

figure();
plot(1:100,reconstruction_error);
xlabel('k');
ylabel('Reconstruction error');
title('Mean reconstruction error as a function of k');

%% QUESTION 2.3 Training and predicting adaboost

for i=1:nimages
    image{i} = imresize(image{i},[19,19]);
end

X_resized = [];
for j=1:length(image)
    X_resized = [X_resized image{j}(:)];  %% making matrix from image column vector
end



Path1a = '/Users/divyadesadla/Desktop/CMU/CMUFall19/MLSP/Homework2/hw2_materials/hw2materials/problem2/train/face';
D=dir(fullfile(Path1a,'*.pgm'));
nimages=length(D);
for i=1:nimages
      train_face_image{i}=double(imread(fullfile(Path1a,D(i).name)));      
end

Path1b = '/Users/divyadesadla/Desktop/CMU/CMUFall19/MLSP/Homework2/hw2_materials/hw2materials/problem2/train/non-face';
D=dir(fullfile(Path1b,'*.pgm'));
nimages=length(D);
for i=1:nimages
      train_non_face_image{i}=double(imread(fullfile(Path1b,D(i).name)));      
end

Path2a = '/Users/divyadesadla/Desktop/CMU/CMUFall19/MLSP/Homework2/hw2_materials/hw2materials/problem2/test/face';
D=dir(fullfile(Path2a,'*.pgm'));
nimages=length(D);
for i=1:nimages
      test_face_image{i}=double(imread(fullfile(Path2a,D(i).name)));      
end

Path2b = '/Users/divyadesadla/Desktop/CMU/CMUFall19/MLSP/Homework2/hw2_materials/hw2materials/problem2/test/non-face';
D=dir(fullfile(Path2b,'*.pgm'));
nimages=length(D);
for i=1:nimages
      test_non_face_image{i}=double(imread(fullfile(Path2b,D(i).name)));      
end




train_face_image_matrix = [];
for j=1:length(train_face_image)
    train_face_image_matrix = [train_face_image_matrix train_face_image{j}(:)];  %% making matrix from image column vector
end


train_non_face_image_matrix = [];
for j=1:length(train_non_face_image)
    train_non_face_image_matrix = [train_non_face_image_matrix train_non_face_image{j}(:)];  %% making matrix from image column vector
end


test_face_image_matrix = [];
for j=1:length(test_face_image)
    test_face_image_matrix = [test_face_image_matrix test_face_image{j}(:)];  %% making matrix from image column vector
end


test_non_face_image_matrix = [];
for j=1:length(test_non_face_image)
    test_non_face_image_matrix = [test_non_face_image_matrix test_non_face_image{j}(:)];  %% making matrix from image column vector
end


%%
X_tr = horzcat(train_face_image_matrix,train_non_face_image_matrix);
X_te = horzcat(test_face_image_matrix,test_non_face_image_matrix);

Y_tr_face = ones(size(train_face_image,2),1);
Y_tr_non_face = ones(size(train_non_face_image,2),1) * -1;
Y_tr = vertcat(Y_tr_face,Y_tr_non_face);

Y_te_face = ones(size(test_face_image,2),1);
Y_te_non_face = ones(size(test_non_face_image,2),1) * -1;
Y_te = vertcat(Y_te_face,Y_te_non_face);


% k=[10,30,50];
% T = [10,50,100,150,200];

%%
T = 10;
for k=10
    final_error = [];
    [U_resized,S_resized,V_resized] = svd(X_resized);
    eigenfaces_resized = U_resized(:,1:k);
    X_tr_proj = pinv(eigenfaces_resized)* X_tr;
    X_te_proj = pinv(eigenfaces_resized)* X_te;
    X_tr_proj = X_tr_proj';
    X_te_proj = X_te_proj';

    model = adaboost_train(X_tr_proj,Y_tr,T);
    Pred = adaboost_predict(model,X_te_proj);
    final_error = [final_error 100*sum(sign(Pred)~=Y_te)/size(Y_te,1)];
end
    
%     E1 = [0.3,0.5,0.7,0.6,0.2,-0.8,0.4,0.2]';
%     E2 = [-0.6,-0.5,-0.1,-0.4,0.4,-0.1,-0.9,0.5]';
%     X_tr = [E1,E2];
%     Y_tr = [1,1,1,1,-1,-1,-1,-1]';
%     
%     F1 = [0.5,0.2,-0.7,0.5,0.1,-0.6,-0.4,-0.8]';
%     F2 = [-0.6,0.5,0.1,-0.2,0.9,-0.3,0.9,-0.8]';
%     X_te = [F1,F2];




%% QUESTION 2.2

function model = adaboost_train(X_tr,Y_tr,T)
model = struct;

[X_tr_sort,index] = sort(X_tr);
Y_tr_sort = Y_tr(index);

weights = ones(size(X_tr,1),1) * 1/length(X_tr);
mov_avg_X_tr = movmean(X_tr_sort,2,1);
Thresh = mov_avg_X_tr(2:end,:);

R_positive = ones(size(X_tr_sort,1),size(X_tr_sort,1)-1);
R_positive(R_positive==triu(R_positive))= -1;
R_negative = R_positive * -1;


all_alpha = [];
threshold = [];
features = [];
signs = [];


for t=1:T
    lowest_error_index = [];
    sign_out = [];
    lowest_error = [];
    
    for f = 1:size(X_tr_sort,2)
        Initial_R_positive = R_positive;  %to keep dummy for manipulations
        Initial_R_negative = R_negative;

        R_positive(R_positive == Y_tr_sort(:,f)) = 0;  %% so as not to add the correctly classified ones while calculating error
        error_1a = - R_positive .* weights .* Y_tr_sort(:,f);
        [error_positive_R error_positive_R_index] = min(sum(error_1a));
        
        
        R_negative(R_negative == Y_tr_sort(:,f)) = 0;  %% so as not to add the correctly classified ones while calculating error
        error_1b = - R_negative .* weights .* Y_tr_sort(:,f);
        [error_negative_R error_negative_R_index] = min(sum(error_1b));
        

        if min([error_positive_R,error_negative_R]) == error_positive_R
            lowest_error = [lowest_error error_positive_R];
            lowest_error_index = [lowest_error_index error_positive_R_index];
            sign_out = [sign_out  1];
        else
            lowest_error = [lowest_error error_negative_R];
            lowest_error_index = [lowest_error_index error_negative_R_index];
            sign_out = [sign_out -1];
        end
        R_positive = Initial_R_positive;
        R_negative = Initial_R_negative;
        
    end
    [lowest_overall_error, lowest_overall_error_index] = min(lowest_error);
    winner_sign = sign_out(lowest_overall_error_index);
    
    if winner_sign==1
        classifier = R_positive(:,lowest_error_index(lowest_overall_error_index));
    else
        classifier = R_negative(:,lowest_error_index(lowest_overall_error_index));
    end
    
    Final_Threshold = Thresh(lowest_error_index(lowest_overall_error_index),lowest_overall_error_index);
    
    Final_alpha = 1/2 * log((1-lowest_overall_error)/lowest_overall_error);
    weights = weights .* exp(-Final_alpha .* Y_tr_sort(:,lowest_overall_error_index) .* classifier);
    Final_feature = lowest_overall_error_index;

    all_alpha = [all_alpha Final_alpha];
    threshold = [threshold Final_Threshold];
    features = [features Final_feature];
    signs = [signs winner_sign];

end 

model.alpha = all_alpha;
model.signs = signs;
model.threshold = threshold;
model.features = features;
end
       



%%
function Pred = adaboost_predict(model,X_te_proj)
alpha = model.alpha;
signs = model.signs;
threshold = model.threshold;
features = model.features;

X_te_proj = X_te_proj(:,features);

compare = bsxfun(@gt,X_te_proj,threshold);
m = compare.*signs;

compare_minus = bsxfun(@le,X_te_proj,threshold);
n = compare_minus.*-signs;

final = m+n;
final_times_alpha = final.*alpha;
Pred = sum(final_times_alpha,2);
end