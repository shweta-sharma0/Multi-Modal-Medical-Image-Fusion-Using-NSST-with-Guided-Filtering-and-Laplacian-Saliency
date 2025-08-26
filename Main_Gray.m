clc; clear; close all;

%% Step 1: Read and Resize Input Images
A = im2double(imread('1(A).tif'));
B = im2double(imread('1(B).tif'));
target_size = [256 256];
A = imresize(A, target_size);
B = imresize(B, target_size);

%% Step 2: NSST Decomposition Parameters
pfilt = 'maxflat';
shear_parameters.dcomp = [4, 4, 4, 4];   % 4 levels with direction splits
shear_parameters.dsize = [8, 8, 16, 32]; % Directional window sizes

r = 5; eps = 0.3; % Guided filtering parameters for low-frequency

%% Step 3: NSST Decomposition
[Y1, shear_f] = nsst_dec2(A, shear_parameters, pfilt);
[Y2, ~]       = nsst_dec2(B, shear_parameters, pfilt);

%% === LOW-FREQUENCY Fusion ===
A_base = guidedfilter(A, Y1{1}, r, eps);
B_base = guidedfilter(B, Y2{1}, r, eps);

base_stack = cat(3, A_base, B_base);
H_base = LapFilter(base_stack);
S_base = GauSaliency(H_base);
W_base = IWconstruct(S_base);
W_base = W_base ./ (sum(W_base, 3) + 1e-12);

% Fusion
Yfused{1} = Y1{1} .* W_base(:,:,1) + Y2{1} .* W_base(:,:,2);

% === Low-Frequency Visualization ===
figure; imshow(A_base, []); title('Guided Filtered A (Low-Freq)');
figure; imshow(B_base, []); title('Guided Filtered B (Low-Freq)');
figure; imshow(H_base(:,:,1), []); title('Laplacian - A Low-Freq');
figure; imshow(H_base(:,:,2), []); title('Laplacian - B Low-Freq');
figure; imshow(S_base(:,:,1), []); title('Saliency - A Low-Freq');
figure; imshow(S_base(:,:,2), []); title('Saliency - B Low-Freq');
figure; imshow(W_base(:,:,1), []); title('Weight - A Low-Freq');
figure; imshow(W_base(:,:,2), []); title('Weight - B Low-Freq');
figure; imshow(Yfused{1}, []); title('Fused Low-Frequency Subband');

%% === HIGH-FREQUENCY Fusion ===
for m = 2:length(Y1) % Levels 2 to N
    [~, ~, d] = size(Y1{m});
    Yfused{m} = zeros(size(Y1{m}));
    
    for n = 1:d % Each directional subband
        A_band = Y1{m}(:,:,n);
        B_band = Y2{m}(:,:,n);
        hf_stack = cat(3, A_band, B_band);

        H_hf = LapFilter(hf_stack);
        S_hf = GauSaliency(H_hf);
        W_hf = IWconstruct(S_hf);
        W_hf = W_hf ./ (sum(W_hf, 3) + 1e-12);

        Yfused{m}(:,:,n) = A_band .* W_hf(:,:,1) + B_band .* W_hf(:,:,2);

        % === Visualization for first few HF subbands ===
        if m <= 3 && n <= 2
            figure; imshow(A_band, []); title(sprintf('Y1{%d}(:,:, %d) - A High-Freq', m, n));
            figure; imshow(B_band, []); title(sprintf('Y2{%d}(:,:, %d) - B High-Freq', m, n));
            figure; imshow(H_hf(:,:,1), []); title('Laplacian - A High-Freq');
            figure; imshow(H_hf(:,:,2), []); title('Laplacian - B High-Freq');
            figure; imshow(S_hf(:,:,1), []); title('Saliency - A High-Freq');
            figure; imshow(S_hf(:,:,2), []); title('Saliency - B High-Freq');
            figure; imshow(W_hf(:,:,1), []); title('Weight - A High-Freq');
            figure; imshow(W_hf(:,:,2), []); title('Weight - B High-Freq');
            figure; imshow(Yfused{m}(:,:,n), []); title('Fused High-Frequency Subband');
        end
    end
end

%% Step 4: NSST Reconstruction
F = nsst_rec2(Yfused, shear_f, pfilt);
F = uint8(F * 255);
figure; imshow(F); title('Final Fused Image');

% Save Result
imwrite(F, 'C:\Users\Shweta Sharma\Desktop\Objective Metrics\05\01.tif');

%% === Helper Functions ===
function H = LapFilter(G)
    L = [1 1 1; 1 -8 1; 1 1 1];
    G = double(G) / 255;
    H = abs(imfilter(G, L, 'replicate'));
end

function S = GauSaliency(H)
    se = fspecial('gaussian', 30, 10);
    S = imfilter(H, se, 'replicate');
    S = S + 1e-12;
    S = S ./ repmat(sum(S, 3), [1 1 size(H, 3)]);
end

function P = IWconstruct(S)
    [~, Labels] = max(S, [], 3);
    [r, c, N] = size(S);
    P = zeros(r, c, N);
    for i = 1:N
        mono = zeros(r, c);
        mono(Labels == i) = 1;
        P(:,:,i) = mono;
    end
end
