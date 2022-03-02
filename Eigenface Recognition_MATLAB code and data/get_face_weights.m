function [weights_of_face] = get_face_weights(im,eignfaces_blk);  
[horizon_pixel,vertical_pixel,basis_number]=size(eignfaces_blk); % Extract number of pixels and basis
im=double(im); % Transfer the 'im' from 'unit8' to 'double'
[horizon_pixel_im,vertical_pixel_im]=size(im); %Extract number of im pixels and basis
weights_of_face=zeros(1,basis_number); % Fabricate new zero set of 'weights_of_face'
    % Calculating the weights_of_face
    for i=1:basis_number; % Extract every image in eigenfaces_blk
        eigenfaces_blk_img=eignfaces_blk(:,:,i); % Number ‘i? image in eigenfaces_blk
        weights_of_face_matrix=eigenfaces_blk_img.*im; % im multiply 'i' image in eigenfaces_blk by point to point
        weights_of_face(i)=sum(sum(weights_of_face_matrix))/(450*300); %Calculate weights_of_face
    end
    %End of calculating the weights_of_face
end

