function [ID] = get_employees_ID_from_DB (im,employees_DB, eignfaces_blk); 

[horizon_pixel,vertical_pixel,basis_number]=size(eignfaces_blk); % Extract number of pixels and basis
im=double(im); % Transfer the 'im' from 'unit8' to 'double'
[horizon_pixel_im,vertical_pixel_im]=size(im); %Extract number of im pixels and basis
match_number=zeros(1,basis_number-1); % Fabricate new zero set of 'Match number in Euclidean_Distance'

% Get weights_of_face
weights_of_face=get_face_weights(im,eignfaces_blk);
% End of getting weights_of_face

% Calculate Euclidean Distance
for i=1:100
    %employees_DB_weights=employees_DB(i).weights; % Extract 'i' weights in employees_DB_weights
    Euclidean_Distance=sqrt(sum((weights_of_face - employees_DB(i).weights).^2)); % Calculate the Euclidean Distance
    match_number(i)=Euclidean_Distance;
end
    [match,ID]=sort(match_number); % Sort the match number from small to big,and the matched ID is smallest 'match' value
    ID_num=ID(1); % Extract smallest number of Euclidean Distance
    ID=employees_DB(ID_num).id; % Extract the id of macthed image
end

