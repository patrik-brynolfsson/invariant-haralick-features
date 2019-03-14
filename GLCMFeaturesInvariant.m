function out = GLCMFeaturesInvariant(glcm, varargin)
% This code calculates gray-level invariant Haralick features according to 
% [4] from one or more GLCMs calculated using e.g. MATLABs graycomatrix()
% function. The GLCMs do not have to be normalized, this is done by the
% function.
% 
% Syntax:
% -------
% featureValues = GLCMFeaturesInvariant(GLCMs, features)
% 
% GLCMs: an m-by-m-by-p array of GLCMs, where m is the dimension of each 
% GLCM and p is the number of GLCMs in the array. Features are calculated
% for each of the p arrays. 
%
% features: a string or cell array of strings, listing the features to 
% calculate. If this is omitted, all features are calculated. 
% 
% GLCMFeaturesInvariant normalizes the GLCMs so the 'volume' of the GLCM is
% equal to 1. This is one step in making the Haralick features gray-level
% invariant. 
%
% Features computed:
% ------------------
% Autocorrelation [2,4]   
% Cluster Prominence [2,4]                   
% Cluster Shade [2,4] 
% Contrast [1,4]                                         
% Correlation [1,4]       
% Difference average
% Difference entropy [1,4] 
% Difference variance [1,4]                   
% Dissimilarity: [2,4]                        
% Energy [1,4]                    
% Entropy [2,4]       
% Homogeneity: (Inverse Difference Moment) [1,2,4] 
% Information measure of correlation1 [1,4]   
% Informaiton measure of correlation2 [1,4]  
% Inverse difference (Homogeneity in matlab): [3,4]     
% Maximum correlation coefficient
% Maximum probability [2,4]                    
% Sum average [1,4]   
% Sum entropy [1,4]  
% Sum of sqaures: Variance [1,4]    
% Sum variance [1,4]   
%
% Example:
% --------
% % First create GLCMs from a 2d image
% GLCMs = graycomatrix(image,'Offset',[0 1; -1 1;-1 0;-1 -1],'Symmetric',...
%   true,'NumLevels',64,'GrayLimits',[0 255]);
% 
% % Sum the GLCMs of different directions to create a direction invariant 
% % GLCM
% GLCM = sum(GLCMs,3)
% 
% % Calculate the invariant Haralick features
% features = GLCMFeaturesInvariant(GLCM)
% 
% % Calulate energy and entropy only
% features = GLCMFeaturesInvariant(GLCM,{'energy','entropy'})
%
% References:
% 1. R. M. Haralick, K. Shanmugam, and I. Dinstein, Textural Features of
% Image Classification, IEEE Transactions on Systems, Man and Cybernetics,
% vol. SMC-3, no. 6, Nov. 1973
% 2. L. Soh and C. Tsatsoulis, Texture Analysis of SAR Sea Ice Imagery
% Using Gray Level Co-Occurrence Matrices, IEEE Transactions on Geoscience
% and Remote Sensing, vol. 37, no. 2, March 1999.
% 3. D A. Clausi, An analysis of co-occurrence texture statistics as a
% function of grey level quantization, Can. J. Remote Sensing, vol. 28, no.
% 1, pp. 45-62, 2002
% 4. Löfstedt T, Brynolfsson P, Asklund T, Nyholm T, Garpebring A (2019) 
% Gray-level invariant Haralick texture features. PLOS ONE 14(2): e0212110.
%
% Started from Avinash Uppupuri's code on Matlab file exchange. It has then
% been vectorized. Three features were not implemented correctly in that
% code, it has since then been changed. The features are: 
%   * Sum of squares: variance
%   * Difference variance
%   * Sum Variance
%
% Written by 
% Patrik Brynolfsson: patrik.brynolfsson@umu.se: 
% Tommy Löfstedt: tommy.lofstedt@umu.se
% Last modified: 2019-02-01

    if nargin == 0
        error('Not enough input arguments')
    else
        if ((size(glcm, 1) <= 1) || (size(glcm, 2) <= 1))
            error('The GLCM should be a 2-D or 3-D matrix.');
        elseif (size(glcm, 1) ~= size(glcm, 2))
            error('Each GLCM should be square with NumLevels rows and NumLevels cols');
        end
    end

    % Handle input parameters
    if nargin < 2
        featureNames = 'all';
    elseif mod(length(varargin), 2) == 1
        featureNames = varargin{1};
        varargin = varargin(2:end);
    end
    input_parser = inputParser();
    checkHomogeneityConstant = @(x) isnumeric(x) && x > 0 && x < inf;
    checkInverseDifferenceConstant = @(x) isnumeric(x) && x > 0 && x < inf;
    addOptional(input_parser, 'homogeneityConstant', 1, checkHomogeneityConstant);
    addOptional(input_parser, 'inverseDifferenceConstant', 1, checkInverseDifferenceConstant);
    parse(input_parser, varargin{:});
    homogeneityConstant = input_parser.Results.homogeneityConstant;
    inverseDifferenceConstant = input_parser.Results.inverseDifferenceConstant;

    % Get size of GLCM
    nGrayLevels = size(glcm, 1);
    nglcm = size(glcm, 3);

    % Differentials
    dA = 1 / (nGrayLevels^2);
    dL = 1 / nGrayLevels;
    dXplusY = 1 / (2 * nGrayLevels - 1);
    dXminusY = 1 / nGrayLevels;
    dkdiag = 1 / nGrayLevels;

    % Normalize the GLCMs
    glcm = bsxfun(@rdivide, glcm, sum(sum(glcm)) * dA);

    % Preallocate
    if scmp('autoCorrelation', featureNames)  % Autocorrelation: [2,4]
        out.autoCorrelation = zeros(1, nglcm);
        features.autoCorrelation = true;
    else
        features.autoCorrelation = false;
    end
    if scmp('clusterProminence', featureNames)  % Cluster Prominence: [2,4]
        out.clusterProminence = zeros(1, nglcm);
        features.clusterProminence = true;
    else
        features.clusterProminence = false;
    end
    if scmp('clusterShade', featureNames)  % Cluster Shade: [2,4]
        out.clusterShade = zeros(1, nglcm);
        features.clusterShade = true;
    else
        features.clusterShade = false;
    end
    if scmp('contrast', featureNames)  % Contrast: matlab/[1,2,4]
        out.contrast = zeros(1, nglcm);
        features.contrast = true;
    else
        features.contrast = false;
    end
    if scmp('correlation', featureNames)  % Correlation: [1,2,4]
        out.correlation = zeros(1, nglcm);
        features.correlation = true;
    else
        features.correlation = false;
    end
    if scmp('differenceAverage', featureNames)
        out.differenceAverage = zeros(1, nglcm);
        features.differenceAverage = true;
    else
        features.differenceAverage = false;
    end
    if scmp('differenceEntropy', featureNames)  % Difference entropy [1,4]
        out.differenceEntropy = zeros(1, nglcm);
        features.differenceEntropy = true;
    else
        features.differenceEntropy = false;
    end
    if scmp('differenceVariance', featureNames)  % Difference variance [1,4]
        out.differenceVariance = zeros(1, nglcm);
        features.differenceVariance = true;
    else
        features.differenceVariance = false;
    end
    if scmp('dissimilarity', featureNames)  % Dissimilarity: [2,4]
        out.dissimilarity = zeros(1, nglcm);
        features.dissimilarity = true;
    else
        features.dissimilarity = false;
    end
    if scmp('energy', featureNames)  % Energy: matlab/[1,2,4]
        out.energy = zeros(1, nglcm);
        features.energy = true;
    else
        features.energy = false;
    end
    if scmp('entropy', featureNames)  % Entropy: [2,4]
        out.entropy = zeros(1, nglcm);
        features.entropy = true;
    else
        features.entropy = false;
    end
    if scmp('homogeneity', featureNames)  % Homogeneity: [2,4] (inverse difference moment)
        out.homogeneity = zeros(1, nglcm);
        features.homogeneity = true;
    else
        features.homogeneity = false;
    end
    if scmp('informationMeasureOfCorrelation1', featureNames)  % Information measure of correlation1 [1,4]
        out.informationMeasureOfCorrelation1 = zeros(1, nglcm);
        features.informationMeasureOfCorrelation1 = true;
    else
        features.informationMeasureOfCorrelation1 = false;
    end
    if scmp('informationMeasureOfCorrelation2', featureNames)  % Informaiton measure of correlation2 [1,4]
        out.informationMeasureOfCorrelation2 = zeros(1, nglcm);
        features.informationMeasureOfCorrelation2 = true;
    else
        features.informationMeasureOfCorrelation2 = false;
    end
    if scmp('inverseDifference', featureNames)  % Homogeneity in matlab
        out.inverseDifference = zeros(1, nglcm);
        features.inverseDifference = true;
    else
        features.inverseDifference = false;
    end
    if scmp('maximalCorrelationCoefficient', featureNames)  % Maximal Correlation Coefficient [1,4]
        out.maximalCorrelationCoefficient = zeros(1, nglcm);
        features.maximalCorrelationCoefficient = true;
    else
        features.maximalCorrelationCoefficient = false;
    end
    if scmp('maximumProbability', featureNames)  % Maximum probability: [2,4]
        out.maximumProbability = zeros(1, nglcm);
        features.maximumProbability = true;
    else
        features.maximumProbability = false;
    end
    if scmp('sumAverage', featureNames)  % Sum average [1,4]
        out.sumAverage = zeros(1, nglcm);
        features.sumAverage = true;
    else
        features.sumAverage = false;
    end
    if scmp('sumEntropy', featureNames)  % Sum entropy [1,4]
        out.sumEntropy = zeros(1, nglcm);
        features.sumEntropy = true;
    else
        features.sumEntropy = false;
    end
    if scmp('sumOfSquaresVariance', featureNames)  % Sum of sqaures: Variance [1,4]
        out.sumOfSquaresVariance = zeros(1, nglcm);
        features.sumOfSquaresVariance = true;
    else
        features.sumOfSquaresVariance = false;
    end
    if scmp('sumVariance', featureNames)  % Sum variance [1,4]
        out.sumVariance = zeros(1, nglcm);
        features.sumVariance = true;
    else
        features.sumVariance = false;
    end
    
    glcmMean = zeros(nglcm, 1);
    uX = zeros(nglcm, 1);
    uY = zeros(nglcm, 1);
    sX = zeros(nglcm, 1);
    sY = zeros(nglcm, 1);

    % pX pY pXplusY pXminusY
    if features.informationMeasureOfCorrelation1 ...
    || features.informationMeasureOfCorrelation2 ...
    || features.maximalCorrelationCoefficient
        pX = zeros(nGrayLevels, nglcm);  % Ng x #glcms[1]
        pY = zeros(nGrayLevels, nglcm);  % Ng x #glcms[1]
    end
    if features.sumAverage ...
    || features.sumVariance ...
    || features.sumEntropy ...
    || features.sumVariance
        pXplusY = zeros((nGrayLevels * 2 - 1), nglcm);  % [1]
    end
    if features.differenceEntropy ...
    || features.differenceVariance
        pXminusY = zeros(nGrayLevels, nglcm);  % [1]
    end
    % HXY1 HXY2 HX HY
    if features.informationMeasureOfCorrelation1
        HXY1 = zeros(nglcm, 1);
        HX = zeros(nglcm, 1);
        HY = zeros(nglcm, 1);
    end
    if features.informationMeasureOfCorrelation2
        HXY2 = zeros(nglcm, 1);
    end

    % Create indices for vectorising code:
    sub = 1:nGrayLevels * nGrayLevels;
    [I, J] = ind2sub([nGrayLevels, nGrayLevels], sub);
    nI = I / nGrayLevels;
    nJ = J / nGrayLevels;

    if features.sumAverage ...
    || features.sumVariance ...
    || features.sumEntropy
        sumLinInd = cell(1, 2 * nGrayLevels - 1);
        for i = 1:2 * nGrayLevels - 1
            diagonal = i - nGrayLevels;
            d = ones(1,nGrayLevels-abs(diagonal));

            diag_ = diag(d, diagonal);
            diag_ud_ = flipud(diag_);
            sumLinInd{i} = find(diag_ud_);
        end
    end
    if features.differenceAverage ...
    || features.differenceVariance ...
    || features.differenceEntropy
        diffLinInd = cell(1,nGrayLevels);
        idx2 = 0:nGrayLevels - 1;
        for i = idx2
            diagonal = i;
            d = ones(1,nGrayLevels - diagonal);
            if (diagonal == 0)
                D = diag(d, diagonal);
                diffLinInd{i+1} = find(D);
            else
                Dp = diag(d, diagonal);
                Dn = diag(d, -diagonal);
                diffLinInd{i+1} = find(Dp + Dn);
            end
        end
    end

    sumIndices = 2:2 * nGrayLevels;

    % Loop over all GLCMs
    for k = 1:nglcm
        currentGLCM = glcm(:, :, k);
        glcmMean(k) = mean2(currentGLCM);

        % For symmetric GLCMs, uX = uY
        uX(k) = sum(nI .* currentGLCM(sub)) * dA;
        uY(k) = sum(nJ .* currentGLCM(sub)) * dA;
        sX(k) = sum((nI - uX(k)).^2 .* currentGLCM(sub)) * dA;
        sY(k) = sum((nJ - uY(k)).^2 .* currentGLCM(sub)) * dA;

        if features.sumAverage ...
        || features.sumVariance ...
        || features.sumEntropy
            for i = sumIndices
                pXplusY(i - 1, k) = sum(currentGLCM(sumLinInd{i-1})) * dkdiag;
            end
        end

        if features.differenceAverage ...
        || features.differenceVariance ...
        || features.differenceEntropy
            idx2 = 0:nGrayLevels - 1;
            for i = idx2
                pXminusY(i + 1, k) = sum(currentGLCM(diffLinInd{i+1})) * dkdiag;
            end
        end

        if features.informationMeasureOfCorrelation1 ...
        || features.informationMeasureOfCorrelation2 ...
        || features.maximalCorrelationCoefficient
            pX(:, k) = sum(currentGLCM, 2) * dL;
            pY(:, k) = sum(currentGLCM, 1)' * dL;
        end
        if features.informationMeasureOfCorrelation1
            HX(k) = -nansum(pX(:, k) .* log(pX(:, k))) * dL;
            HY(k) = -nansum(pY(:, k) .* log(pY(:, k))) * dL;
            HXY1(k) = -nansum(currentGLCM(sub)' .* log(pX(I, k) .* pY(J, k))) * dA;
        end
        if features.informationMeasureOfCorrelation2
            HXY2(k) = -nansum(pX(I, k) .* pY(J, k) .* log(pX(I, k) .* pY(J, k))) * dA;
        end

        % Haralick features:
        % -----------------
        if features.energy
            out.energy(k) = sum(currentGLCM(sub).^2) * dA;
        end
        if features.contrast
            out.contrast(k) = sum((nI - nJ).^2 .* currentGLCM(sub)) * dA;
        end

        if features.autoCorrelation ...
        || features.correlation
            autoCorrelation = sum(nI .* nJ .* currentGLCM(sub)) * dA;
            if features.autoCorrelation
                out.autoCorrelation(k) = autoCorrelation;
            end
        end

        if features.correlation
            if sX(k) < eps || sY(k) < eps
                out.correlation(k) = min(max((autoCorrelation - uX(k) .* uY(k)),-1),1);
            else
                out.correlation(k) = (autoCorrelation - uX(k) .* uY(k)) ./ sqrt(sX(k) .* sY(k));
            end
        end
        if features.sumOfSquaresVariance
            out.sumOfSquaresVariance(k) = sum(currentGLCM(sub) .* ((nI - uX(k)).^2)) * dA;
        end
        if features.homogeneity
            out.homogeneity(k) = sum(currentGLCM(sub) ./ (1 + homogeneityConstant * (nI - nJ).^2)) * dA;
        end
        if features.sumAverage ...
        || features.sumVariance
            sumAverage = sum(bsxfun(@times, ((2 * (sumIndices - 1)) / (2 * nGrayLevels - 1))', pXplusY(sumIndices - 1, k))) * dXplusY;
            if features.sumAverage
                out.sumAverage(k) = sumAverage;
            end
        end
        if features.sumVariance
            out.sumVariance(k) = sum((((2 * (sumIndices - 1)) / (2 * nGrayLevels - 1)) - sumAverage)'.^2 .* pXplusY(sumIndices - 1, k)) * dXplusY;
        end
        if features.sumEntropy
            out.sumEntropy(k) = -nansum(pXplusY(sumIndices - 1, k) .* log(pXplusY(sumIndices - 1, k))) * dXplusY;  % Differential entropy
        end
        if features.entropy ...
        || features.informationMeasureOfCorrelation1 ...
        || features.informationMeasureOfCorrelation2
            entropy = -nansum(currentGLCM(sub) .* log(currentGLCM(sub))) * dA;  % Differential entropy
            if features.entropy
                out.entropy(k) = entropy;
            end
        end

        if features.differenceAverage ...
        || features.differenceVariance
            differenceAverage = sum(bsxfun(@times, ((idx2 + 1) / nGrayLevels)', pXminusY(idx2 + 1, k))) * dXminusY;
            if features.differenceAverage
                out.differenceAverage(k) = differenceAverage;
            end
        end

        if features.differenceVariance
            out.differenceVariance(k) = sum((((idx2 + 1) / nGrayLevels) - differenceAverage).^2' .* pXminusY(idx2 + 1, k)) * dXminusY;
        end
        if features.differenceEntropy
            out.differenceEntropy(k) = -nansum(pXminusY(idx2 + 1, k) .* log(pXminusY(idx2 + 1, k))) * dXminusY;  % Differential entropy
        end
        if features.informationMeasureOfCorrelation1
            infoMeasure1 = (entropy - HXY1(k)) ./ (max(HX(k), HY(k)));
            out.informationMeasureOfCorrelation1(k) = infoMeasure1;
        end
        if features.informationMeasureOfCorrelation2
            infoMeasure2 = sqrt(1 - exp(-2 * (HXY2(k) - entropy)));
            out.informationMeasureOfCorrelation2(k) = infoMeasure2;
        end
        if features.maximalCorrelationCoefficient

            % Correct by eps if the matrix has columns or rows that sums to zero.
            P = currentGLCM;
            pX_ = pX(:, k);
            if any(pX_ < eps)
                pX_ = pX_ + eps;
                pX_ = pX_ / (sum(pX_(:)) * dL);
            end
            pY_ = pY(:, k);
            if any(pY_ < eps)
                pY_ = pY_ + eps;
                pY_ = pY_ / (sum(pY_(:)) * dL);
            end

            % Compute the Markov matrix
            Q = zeros(size(P));
            for i = 1:nGrayLevels
                Pi = P(i, :);
                pXi = pX_(i);
                for j = 1:nGrayLevels
                    Pj = P(j, :);
                    d = pXi * pY_.';
                    if d < eps
                        fprintf('Division by zero in the maximalCorrelationCoefficient!\n');
                    end
                    Q(i, j) = dA * sum((Pi .* Pj) ./ d);
                end
            end

            % Compute the second largest eigenvalue
            if any(isinf(Q))
                e2 = NaN;
            else
                try
                    E = eigs(Q, 2);
                catch
                    try
                        E = eig(Q);
                    catch
                        fprintf('Could not compute the maximalCorrelationCoefficient!\n');
                    end
                end

                % There may be a near-zero imaginary component here
                if isreal(E(1)) && isreal(E(2))
                    e2 = E(2);
                else
                    e2 = min(real(E(1)), real(E(2)));
                end
            end

            out.maximalCorrelationCoefficient(k) = e2;
        end

        if features.dissimilarity
            dissimilarity = sum(abs(nI - nJ) .* currentGLCM(sub)) * dA;
            out.dissimilarity(k) = dissimilarity;
        end
        if features.clusterShade
            out.clusterShade(k) = sum((nI + nJ - uX(k) - uY(k)).^3 .* currentGLCM(sub)) * dA;
        end
        if features.clusterProminence
            out.clusterProminence(k) = sum((nI + nJ - uX(k) - uY(k)).^4 .* currentGLCM(sub)) * dA;
        end
        if features.maximumProbability
            out.maximumProbability(k) = max(currentGLCM(:));
        end
        if features.inverseDifference
            out.inverseDifference(k) = sum(currentGLCM(sub) ./ (1 + inverseDifferenceConstant * abs(nI - nJ) )) * dA;
        end
    end
end

% GLCM Features (Soh, 1999; Haralick, 1973; Clausi 2002)
%  f1. Angular Second Moment / Energy / Uniformity
%  f2. Contrast / Inertia
%  f3. Correlation
%  f4. Sum of Squares: Variance
%  f5. Inverse Difference Moment / Homogeneity
%  f6. Sum Average
%  f7. Sum Variance
%  f8. Sum Entropy
%  f9. Entropy
% f10. Difference Variance
% f11. Difference Entropy
% f12. Information Measure of Correlation 1
% f13. Information Measure of Correlation 2
% f14. Maximal Correlation Coefficient
% f15. Autocorrelation
% f16. Cluster Shade
% f17. Cluster Prominence
% f18. Maximum Probability
% f19. Inverse Difference
% f20. Difference Average

function contains = scmp(string, list)

    contains = any(strcmpi(list, string)) || any(strcmpi(list, 'all'));

end