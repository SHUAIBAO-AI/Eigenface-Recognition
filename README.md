# Eigenface-Recognition


Orthonormal basis
	These eigenfaces are orthogonal.
	If we want to confirm two matrices are orthogonal, one can indeed consider matrices as vectors; an n×n matrix is then just a vector in an n^2 dimensional vector space. In such a vector space, one can then define a scalar product just as in any other vector space. It turns out that for real matrices, the standard scalar product can be expressed in the simple form ⟨A,B⟩=tr(ABT) , and thus we can also define two matrices as orthogonal to each other when ⟨A,B⟩=0, j
ust as with any other vector space.
	So, in our task, to prove the matrix are orthogonal for each other, we need to extract every matrix in this database and multiply them with each other. After we extract the matrix and multiply them, we found that the sum of these product is 0, and in MATLAB its not exactly 0, its -2*10^-8, its very close to 0. So that we can confirm that the matrix are orthogonal.
	The related results in MATLAB attached to Figure1, ’sum_sum’ is the result.

Code:
%% Task1 Orthonormal basis
clear all;
clc;
load('E:\2019\412image processinglab\Lab3\lab3-Material\data_for_labC.mat');%get data from data_for_labC.mat
[horizon_pixel,vertical_pixel,basis_number]=size(eignfaces_blk); % Extract number of pixels and basis
sum_initial=0;
basis_number=basis_number-2;
basis_number_1=basis_number-1
for i=1:basis_number;i_1=i+1;
for j=i_1:basis_number_1
orthonormal_sum=eignfaces_blk(:,:,i).*eignfaces_blk(:,:,j); %Extract each figure and find if they are Orthonormal
sum_num=sum_initial+sum(sum(orthonormal_sum)); %Calculate the orthonormal number,sum_num is -2.626858730536696e-08, approximately zero,so that it's Orthonormal basis
end
End
 
Output：
![image](https://github.com/STPChenFang/Eigenface-Recognition/blob/main/IMG_Eigenface%20recoginition/image001.png)

Figure1: Task1 Orthonormal basis


Evaluating the Eigenfaces weights of a face.
	If we want to calculate the weights of image, whose database is eignfaces_blk, we should make it clear that what is eignfaces and how to show it.
	At first, what is eignfaces? Eigenfaces is the name given to a set of eigenvectors when they are used in the computer vision problem of human face recognition. The approach of using eigenfaces for recognition was developed by Sirovich and Kirby (1987) and used by Matthew Turk and Alex Pentland in face classification. The eigenvectors are derived from the covariance matrix of the probability distribution over the high-dimensional vector space of face images. The eigenfaces themselves form a basis set of all images used to construct the covariance matrix. This produces dimension reduction by allowing the smaller set of basis images to represent the original training images. Classification can be achieved by comparing how faces are represented by the basis set.[1]
	Secondly, in eignface recognition, transform domain are very important concept, two basic factors of the ransform domain are basis and coefficients. Basis for the image space is the number of images of the basis is equal to the number of pixels in the images, and all of the basis is orthogonal for each other.

Code:
function [weights_of_face] = get_face_weights(im,eignfaces_blk);  
[horizon_pixel,vertical_pixel,basis_number]=size(eignfaces_blk); % Extract number of pixels and basis
im=double(im); % Transfer the 'im' from 'unit8' to 'double'
[horizon_pixel_im,vertical_pixel_im]=size(im); %Extract number of im pixels and basis
weights_of_face=zeros(1,basis_number); % Fabricate new zero set of 'weights_of_face'
    % Calculating the weights_of_face
    for i=1:basis_number; % Extract every image in eigenfaces_blk
        eigenfaces_blk_img=eignfaces_blk(:,:,i); % Number ‘i’ image in eigenfaces_blk
        weights_of_face_matrix=eigenfaces_blk_img.*im; % im multiply 'i' image in eigenfaces_blk by point to point
        weights_of_face(i)=sum(sum(weights_of_face_matrix)); %Calculate weights_of_face
    end
    %End of calculating the weights_of_face
End

clear all;
clc;
load('C:\Users\Documents\Lab3\data_for_labC.mat');%get data from data_for_labC.mat
im=imread('C:\Users\Shuai.bao19\Documents\Lab3\find_id.jpg'); %Get 'findID' image data
 
% Get weights_of_face
weights_of_face=get_face_weights(im,eignfaces_blk);
% 
Output：
![image](https://github.com/STPChenFang/Eigenface-Recognition/blob/main/IMG_Eigenface%20recoginition/image002.png)
Figure2: Task2: Evaluating the Eigenfaces weights of a face.

Task3: Face generation from its “weights”
Answer: 
	In this function, we input image ‘weights’ to generate the matched face from the set.
	In eignface protocol, any human face can be regarded as a combination of some “standard faces”, we call it as eigen-faces, and the ‘weights’ to describe that someone face might be composed of the  “average face” plus 5% from eigenface 1,0%from eigenface 2, and so on.
	So in this task, we want to generate a face from its ‘weights’, we need to multiply each weights to every standard face in database which is ‘eigenfaces_blk’, in MATLAB, its that we should multiply the ‘weights’ number to matrix, and then we add them together, the sum of image is what we need.
	
Code:
%% Task3 Face generation from its “weights”
clear all;
clc;
load('C:\Users\Documents\Lab3\data_for_labC.mat');%get data from data_for_labC.mat
im=imread('C:\Users\Documents\Lab3\find_id.jpg'); %Get 'findID' image data
im_original=im; % Keep original image
% Get weights_of_face
weights_of_face=get_face_weights(im,eignfaces_blk);
% End of getting weights_of_face
 
% Get im by 'generate_face_from_weights'
im=generate_face_from_weights(weights_of_face,eignfaces_blk); %
% End of getting im by 'generate_face_from_weights'
 
subplot(1,2,1);imshow(im_original,[]);title('Original image');% Show the original image
subplot(1,2,2);imshow(im,[]);title('After calculation image');% Show the image after calculation
Subfunction：
function [im] = generate_face_from_weights(weights_of_face,eignfaces_blk)
[horizon_pixel,vertical_pixel,basis_number]=size(eignfaces_blk); % Extract number of pixels and basis
im=zeros(horizon_pixel,vertical_pixel); % Fabricate new zero 'weights_of_face' matrix
im=double(im); % Transfer the 'im' from 'unit8' to 'double'
    % Calculating the weights_of_face face image
    for i=1:basis_number; % Extract every image in eigenfaces_blk
        eigenfaces_blk_img=eignfaces_blk(:,:,i); % Number ‘i’ image in eigenfaces_blk
        weights_of_face_img=weights_of_face(i)*eigenfaces_blk_img; % Multiply each weights to basis image 
        im=im+weights_of_face_img; % Add the basis image to im
    end
    %End of calculating the weights_of_face face image
end

Output:

![image](https://github.com/STPChenFang/Eigenface-Recognition/blob/main/IMG_Eigenface%20recoginition/image004.png)

Figure3 Face generation from its “weights”

Task4: Recognizing an employee from his/her image.
Answer: 
	After we got the ‘weights’ of ‘im’, in this task it’s a vector that contain 100 number which describe the weights of eigenfaces, and in our database employees_DB, every employee’s weights number also stored in this employees_DB, what we need to do is that we should find out the matched vector in employees_DB, which means ,we need to find the match vector in high dimension space.
	To find the most match vector, we need to use Euclidean Distance between two vectors to determine whether the distance is 0 or not, so that we calculate each distance between  ‘weights’ of ‘im’ and each vector in employees_DB. What we found is 96, employee’s ID is 96.
	Because in our calculation, ID 96 has the smallest Euclidean Distance, and in our confirmation, Figure3 also has the same image with original image.

Code:
%% Task4 Recognizing an employee from his/her image. 
clear all;
clc;
load('C:\Users\Shuai.bao19\Documents\Lab3\data_for_labC.mat');%get data from data_for_labC.mat
im=imread('C:\Users\Shuai.bao19\Documents\Lab3\find_id.jpg'); %Get 'find_id.jpg' image data
 
ID = get_employees_ID_from_DB (im,employees_DB, eignfaces_blk); 
Subfunction：
function [ID] = get_employees_ID_from_DB (im,employees_DB, eignfaces_blk); 
 
[horizon_pixel,vertical_pixel,basis_number]=size(eignfaces_blk); % Extract number of pixels and basis
im=double(im); % Transfer the 'im' from 'unit8' to 'double'
[horizon_pixel_im,vertical_pixel_im]=size(im); %Extract number of im pixels and basis
match_number=zeros(1,basis_number); % Fabricate new zero set of 'Match number in Euclidean_Distance'
 
% Get weights_of_face
weights_of_face=get_face_weights(im,eignfaces_blk);
% End of getting weights_of_face
 
% Calculate Euclidean Distance
for i=1:(basis_number-1);
    %employees_DB_weights=employees_DB(i).weights; % Extract 'i' weights in employees_DB_weights
    Euclidean_Distance=sqrt(sum((weights_of_face-employees_DB(i).weights).^2)); % Calculate the Euclidean Distance
    match_number(i)=Euclidean_Distance;
end
    [match,ID]=sort(match_number); % Sort the match number from small to big,and the matched ID is smallest 'match' value
    ID_num=ID(1); % Extract smallest number of Euclidean Distance
    ID=employees_DB(ID_num).id; % Extract the id of macthed image
end

Task5: Discussion.
Answer: 
	As an appearance-based approach, eigenface recognition method has several advantages:
 	1) Raw intensity data are used directly for learning and recognition without any significant low-level or mid-level processing; 
(2) No knowledge of geometry and reflectance of faces is required; 
(3) Data compression is achieved by the low-dimensional subspace representation; 
(4) Recognition is simple and efficient compared to other matching approaches.
	Apart from those advantages, eigenface recognition also has several disadvantages: 
First, learning is very time-consuming,
Secondly, its recognition rate decreases for recognition under varying pose and illumination. 
Third, eigenface approach is not robust when dealing with extreme variations in pose as well as in expression and disguise. 
Fourth, the method may require uniform background which may not be satisfied in most natural scenes.
The question we faced now: illumination, posture, shield, change of age, the quality of image, lacking of samples, database size. All of these factors can affect the recognition.
About the robustness of the eigenface-based image recognition, in different conditions, for example, if the image has salt&pepper noise, it will has influence on our recognition, if the salt&pepper ratios is not so high, for example, the ratios of noise is lower than 20%, we can still figure out the weights and calculate the Euclidean Distance to match the ID in database. Of cause its not so exactly that the ratio must below 20%, we need to calculate the influence in different circumstance.
How can we fabricate more stable calculation? For example, if the image is not exist in database, which means the Euclidean Distance we calculated will never be 0, or lower than threshold. So we need to design a new protocol to deny the image before we started to recognize the image.
And what if we find the image is full of noise, such as Gaussian White Noise or salt&pepper noise? Or, if we cannot get the ideal image, there are too much black or white blocks on image so that we cannot figure out it? Maybe we should wipe the noise before we recognize it.
Eigenfaces can match the ideal faces in database. But it will cost too much time, and its recognition rate is not so high in low illumination.
Apart from Eigenfaces, we also have different algorithm to recognize the face, Local Binary Patterns(LBP) and Fisherface.
	Eigenfaces use PCA analysis, but in PCA, the biggest obstacle is incremental learning problems. With the increase of samples, it is necessary to keep the dimension of subspace unchanged, so the accuracy of this method is a little bit bad.

References:
[1] https://en.wikipedia.org/wiki/Eigenface 
