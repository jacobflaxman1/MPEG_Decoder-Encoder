clear; clc; close all;


% initialize zeros matrices, constants, and function handles

videoFile = 'walk_qcif.avi';

vid = VideoReader(videoFile);
mbSize = 16;  
searchArea = 16; 
height = 144;
width = 176;
Y_quantize = [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62; 18 22 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
cbcr_quantize = [17 18 24 47 99 99 99 99; 18 21 26 66 99 99 99 99; 24 26 56 99 99 99 99 99; 47 66 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99];


dct_handle = @(block_struct) dct2(block_struct.data);
idct_handle = @(block_struct) idct2(block_struct.data);
quantize_handle = @(block_struct, q_matrix) round(block_struct.data ./ q_matrix);
dequantize_handle = @(block_struct, q_matrix) block_struct.data .* q_matrix;

frames = zeros(144, 176, 3, 5);  
encoded_I_Y = zeros(height, width, 1); 
encoded_I_Cb = zeros(height / 2, width / 2, 1);
encoded_I_Cr = zeros(height / 2, width / 2, 1);
encoded_P_Y = zeros(height, width, 4);
encoded_P_Cb = zeros(height / 2, width / 2, 4);
encoded_P_Cr = zeros(height / 2, width / 2, 4);

%
%       MAIN LOOP BEGIN
%
%

p_index = 1;  
motion_vectors_store = zeros(height / mbSize, width / mbSize, 2, 4);  
for i = 1:5
    frameRGB = readFrame(vid);  
    ycbcr = rgb2ycbcr(frameRGB);  

    if i == 1  % grab the first frame as the i frame and encode it for storage and reference
        [Y_encoded, Cb_recon, Cr_recon] = process_I_frame(ycbcr, Y_quantize, cbcr_quantize);
        encoded_I_Y(:, :, 1) = Y_encoded;
        encoded_I_Cb(:, :, 1) = Cb_recon;
        encoded_I_Cr(:, :, 1) = Cr_recon;
    else  % next 4 frames are p frames -> process them by
        % motion est -> dct -> quant -> diff img -> storage
        [Y_diff_quant, Cb_subsampled, Cr_subsampled, motion_vectors] = process_P_frame(encoded_I_Y(:, :, 1), ycbcr, mbSize, searchArea, Y_quantize);
        encoded_P_Y(:, :, i - 1) = Y_diff_quant;
        encoded_P_Cb(:, :, i - 1) = Cb_subsampled;
        encoded_P_Cr(:, :, i - 1) = Cr_subsampled;
        motion_vectors_store(:, :, :, i - 1) = motion_vectors;  % store the motion vectors for reconstruction at decode
            
        
        figure;
        imshow(Y_diff_quant);  
        colormap gray; 
        axis image off;  
        title(sprintf('Diff Frame %d', i));
  


        figure; 
        [rows, cols] = size(encoded_I_Y(:, :, 1));
        [X, Y] = meshgrid(1:mbSize:cols, 1:mbSize:rows);
        U = motion_vectors(:, :, 1);
        V = motion_vectors(:, :, 2);
        quiver(X, Y, U, V, 'AutoScale', 'on', 'Color', 'b');  
        hold off; 
        axis tight;
    end
end


% decode frames 
    % returns all of the decoded frames
[decoded_frames] = decode(encoded_I_Y, encoded_P_Y, encoded_I_Cb, encoded_I_Cr, encoded_P_Cb, encoded_P_Cr, Y_quantize, cbcr_quantize, mbSize, motion_vectors_store);

for i = 1:size(decoded_frames, 4)
    figure;
    imshow(decoded_frames(:, :, :, i));
    title(['Reconstructed Frame ', num2str(i)]);
end


%
%       MAIN LOOP END
%
%







%
%
%   FUNCTIONS
%
%



function [Y_recon, Cb_recon, Cr_recon] = process_I_frame(iframe_ycbcr, Y_quantize, cbcr_quantize)
    Y = iframe_ycbcr(:, :, 1);
    Cb = iframe_ycbcr(:, :, 2);
    Cr = iframe_ycbcr(:, :, 3);
    Cb_subsample = Cb(1:2:end, 1:2:end);
    Cr_subsample = Cr(1:2:end, 1:2:end);

    % start with dct and quantize
    Y_dct = blockproc(double(Y), [8 8], @(block_struct) dct2(block_struct.data));
    Cb_dct = blockproc(double(Cb_subsample), [8 8], @(block_struct) dct2(block_struct.data));
    Cr_dct = blockproc(double(Cr_subsample), [8 8], @(block_struct) dct2(block_struct.data));
    Y_quantized = blockproc(Y_dct, [8 8], @(block_struct) round(block_struct.data ./ Y_quantize));
    Cb_quantized = blockproc(Cb_dct, [8 8], @(block_struct) round(block_struct.data ./ cbcr_quantize));
    Cr_quantized = blockproc(Cr_dct, [8 8], @(block_struct) round(block_struct.data ./ cbcr_quantize));

    % immediately inv quantize and inverse dct
    Y_dequantized = blockproc(Y_quantized, [8 8], @(block_struct) block_struct.data .* Y_quantize);
    Cb_dequantized = blockproc(Cb_quantized, [8 8], @(block_struct) block_struct.data .* cbcr_quantize);
    Cr_dequantized = blockproc(Cr_quantized, [8 8], @(block_struct) block_struct.data .* cbcr_quantize);
    Y_recon = blockproc(Y_dequantized, [8 8], @(block_struct) idct2(block_struct.data));
    Cb_recon = blockproc(Cb_dequantized, [8 8], @(block_struct) idct2(block_struct.data));
    Cr_recon = blockproc(Cr_dequantized, [8 8], @(block_struct) idct2(block_struct.data));
    % returns reconstructed I frame 
end


function [Y_diff_quant, Cb_subsampled, Cr_subsampled, motion_vectors] = process_P_frame(iframe_ycbcr, pframe_ycbcr, mbSize, searchArea, Y_quantize)
    % function to process P frames 
    % returns single encoded p frame

    Y_pframe = double(pframe_ycbcr(:, :, 1));
    Cb_pframe = pframe_ycbcr(:, :, 2);
    Cr_pframe = pframe_ycbcr(:, :, 3);

    % 4:2:0 subsampking 
    Cb_subsampled = Cb_pframe(1:2:end, 1:2:end);
    Cr_subsampled = Cr_pframe(1:2:end, 1:2:end);

    [motion_vectors, diffFrame_Y] = exhaustive_search(double(iframe_ycbcr(:, :, 1)), Y_pframe, mbSize, searchArea);

    % dct and quantize the diff frame to send to decoder 
    diffFrame_dct_Y = blockproc(diffFrame_Y, [8 8], @(block_struct) dct2(block_struct.data));
    Y_diff_quant = blockproc(diffFrame_dct_Y, [8 8], @(block_struct) round(block_struct.data ./ Y_quantize));   
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Frame Reconstruction                     %
%                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function reconFrame = reconstructFrame(iframe, motionVectors, diffFrame, mbSize)
    [rows, cols] = size(iframe); % get size of the i frame
    reconFrame = zeros(size(iframe)); % initialize the matrix for the reconstructed frame
    
    for i = 1:mbSize:rows-mbSize+1 % move across the motion vector matrix one macro block at a time
        for j = 1:mbSize:cols-mbSize+1 % move vertically across the motion vector matrix
            dx = motionVectors(floor(i/mbSize)+1, floor(j/mbSize)+1, 1);
            % incrementing by mbsize then dividing the index by mbSize 
            % to 
            dy = motionVectors(floor(i/mbSize)+1, floor(j/mbSize)+1, 2);
            reconFrame(i:i+mbSize-1, j:j+mbSize-1) = iframe(i+dx:i+dx+mbSize-1, j+dy:j+dy+mbSize-1);
        end
    end
    
    reconFrame = reconFrame + diffFrame;
   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Exhaustive Search                    %
%                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [motionVectors, diffFrame] = exhaustive_search(iframe, pframe, mbSize, searchArea)
    [rows, cols] = size(iframe); % get the size of the I frame
    motionVectors = zeros(floor(rows/mbSize), floor(cols/mbSize), 2); % initialize the motion vector matrix
    predFrame = zeros(size(iframe));    % initialize the predicted frame to zeros
    for i = 1:mbSize:rows-mbSize+1 % move one macro block at a time across each row
        for j = 1:mbSize:cols-mbSize+1 % move one macro block at a time across each column
            bestSAD = inf; % initialize the bestSad to inf as we are looking for the lowest SAD
            for x = max(i-searchArea, 1):min(i+searchArea, rows-mbSize+1) 
                    % x represents the top horizontal axis on the macroblock starting on the far left 
                    % had to use max and min to ensure the cursor doesn't
                    % start outside or go outside the search area if its
                    % out of bounds
                for y = max(j-searchArea, 1):min(j+searchArea, cols-mbSize+1)
                    % y represents the vertical axis starting at the top
                    % combining the x and y we are starting at the top left
                    % of each macroblock

                    % calculate the SAD for the current search against the
                    % iframe
                   
                    currentMb = double(pframe(i:i+mbSize-1, j:j+mbSize-1));
                    candidateMb = double(iframe(x:x+mbSize-1, y:y+mbSize-1));
                    
                    % calculate the SAD using the curent macro block and
                    % the I frame reference macro block
                    SAD = sum(abs(currentMb - candidateMb), 'all');                    
                    if SAD < bestSAD
                        bestSAD = SAD;
                        dx = x - i; % the delta on the x axis against the i frame
                        dy = y - j; % the delta on the y axis against the i frame
                    end
                end
            end

            % append the motion vector into the matrice of motion vectors,
            motionVectors(floor(i/mbSize)+1, floor(j/mbSize)+1, :) = [dx, dy]; 

            % reconstruct the P frame from the motion vecotrs
            % grab the macro block at the i frame and add the dx and dy to
            % the x and y. This whill give us the predicted frame
            predFrame(i:i+mbSize-1, j:j+mbSize-1) = iframe(i+dx:i+dx+mbSize-1, j+dy:j+dy+mbSize-1);

           
        end
    end
    % calculate the difference frame by subtracting the p frame from the
    % predicted frame 
    diffFrame = pframe - predFrame;

end



function decoded_frames = decode(encoded_I_Y, encoded_P_Y, encoded_I_Cb, encoded_I_Cr, encoded_P_Cb, encoded_P_Cr, Y_quantize, cbcr_quantize, mbSize, motion_vectors)
    % function to decode the frames sent byu encoder 


    num_frames = 1 + size(encoded_P_Y, 3); % I frame plus all fo the p frames we encoded 
    height = size(encoded_I_Y, 1);
    width = size(encoded_I_Y, 2);
    decoded_frames = zeros(height, width, 3, num_frames, 'uint8');

    % start by upsammpling the I frame and adding it to buffer
    Y_I = uint8(encoded_I_Y);
    Cb_I = imresize(uint8(encoded_I_Cb), [height, width], 'bicubic');
    Cr_I = imresize(uint8(encoded_I_Cr), [height, width], 'bicubic');
    decoded_frames(:, :, :, 1) = ycbcr2rgb(cat(3, Y_I, Cb_I, Cr_I));

    % then decode p frames by -> inv quant and dct -> frame reconstruction using diff frame and
    % motion vectors-> upsample cb and cr -> concat then add to buffer
    for i = 1:size(encoded_P_Y, 3)
        ref_frame_Y = double(decoded_frames(:, :, 1, i)); 
        diff_Y = double(encoded_P_Y(:, :, i));  
        mv = motion_vectors(:, :, :, i);  
     
        Y_recon = uint8(reconstructFrame(ref_frame_Y, mv, diff_Y, mbSize)); % Convert the result to uint8 after reconstruction

        Cb_recon = imresize(uint8(encoded_P_Cb(:, :, i)), [height, width], 'bicubic');
        Cr_recon = imresize(uint8(encoded_P_Cr(:, :, i)), [height, width], 'bicubic');

        decoded_frames(:, :, :, i + 1) = ycbcr2rgb(cat(3, Y_recon, Cb_recon, Cr_recon));
    end
end


