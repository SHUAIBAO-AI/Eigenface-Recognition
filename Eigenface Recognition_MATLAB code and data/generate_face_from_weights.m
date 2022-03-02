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

