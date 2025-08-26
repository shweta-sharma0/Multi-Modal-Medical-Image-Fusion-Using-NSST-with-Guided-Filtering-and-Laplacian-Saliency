clc; clear; close all;

%% Step 1: Read and Resize Input Images
A = im2double(imread('5(A).png'));  % CT image
B = im2double(imread('5(B).png'));  % PET/SPECT image
target_size = [256 256];
A = imresize(A, target_size);
B = imresize(B, target_size);

% Convert grayscale to RGB if needed
if size(A, 3) == 1, A = repmat(A, [1 1 3]); end
if size(B, 3) == 1, B = repmat(B, [1 1 3]); end

% Convert to grayscale for NSST fusion
A_gray = rgb2gray(A);
B_gray = rgb2gray(B);

%% Step 2: NSST Parameters
pfilt = 'maxflat';
shear_parameters.dcomp = [4, 4, 4, 4];   % Levels
shear_parameters.dsize = [8, 8, 16, 32]; % Directions
r = 5; eps = 0.3; % Guided filter params

%% Step 3: NSST Decomposition
[Y1, shear_f] = nsst_dec2(A_gray, shear_parameters, pfilt);
[Y2, ~]       = nsst_dec2(B_gray, shear_parameters, pfilt);

%% Step 4: Low-Frequency Fusion
A_base = guidedfilter(A_gray, Y1{1}, r, eps);
B_base = guidedfilter(B_gray, Y2{1}, r, eps);
base_stack = cat(3, A_base, B_base);

H_base = LapFilter(base_stack);
S_base = GauSaliency(H_base);
W_base = IWconstruct(S_base);
W_base = W_base ./ (sum(W_base, 3) + 1e-12);
Yfused{1} = Y1{1} .* W_base(:,:,1) + Y2{1} .* W_base(:,:,2);

%% Step 5: High-Frequency Fusion
for m = 2:length(Y1)
    [~, ~, d] = size(Y1{m});
    Yfused{m} = zeros(size(Y1{m}));
    for n = 1:d
        A_band = Y1{m}(:,:,n);
        B_band = Y2{m}(:,:,n);
        hf_stack = cat(3, A_band, B_band);
        H_hf = LapFilter(hf_stack);
        S_hf = GauSaliency(H_hf);
        W_hf = IWconstruct(S_hf);
        W_hf = W_hf ./ (sum(W_hf, 3) + 1e-12);
        Yfused{m}(:,:,n) = A_band .* W_hf(:,:,1) + B_band .* W_hf(:,:,2);
    end
end

%% Step 6: Reconstruction
F_gray = nsst_rec2(Yfused, shear_f, pfilt);
F_gray = mat2gray(F_gray);

%% Step 7: Color Mapping — HSV or Pseudo-Color
if size(B, 3) == 3 && ~isequal(B(:,:,1), B(:,:,2))  % PET is pseudo-colored
    % Use PET (B) HSV as base
    B_hsv = rgb2hsv(B);
    B_hsv(:,:,3) = F_gray;  % Replace Value
    F_color = hsv2rgb(B_hsv);
else
    % Apply pseudo-coloring
    F_color = ind2rgb(gray2ind(F_gray, 256), jet(256));
end

%% Step 8: Display + Save
figure; imshow(F_color); title('Fused Color Image (PET-handled)');
imwrite(F_color, 'C:\Users\Shweta Sharma\Desktop\Objective Metrics\05\05.tif');

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
