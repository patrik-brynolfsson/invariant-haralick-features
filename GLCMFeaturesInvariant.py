#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:59:23 2021

Copyright (c) 2018-2021, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import numpy as np
import scipy.sparse.linalg

__all__ = ["GLCMFeaturesInvariant"]

MACHINE_EPSILON = np.finfo(np.float64).eps


def GLCMFeaturesInvariant(glcms,
                          props=None,
                          homogeneity_constant=1.0,
                          inverse_difference_constant=1.0):
    r"""Compute gray-level invariant Haralick features.

    This function computes gray-level invariant Haralick features according
    to [4] from one or more GLCMs computed using e.g. scikit-image's
    greycomatrix function. The GLCMs do not have to be normalized, this is done
    by the function.

    GLCMFeaturesInvariant normalizes the GLCMs so the 'volume' of the GLCM is
    equal to 1. This is one step in making the Haralick features gray-level
    invariant.

    This implementation is a port of our own Matlab implementation, which was
    based on Avinash Uppupuri's code on Matlab file exchange.

    Parameters
    ----------
    GLCMs : numpy array, shape m-by-m-by-p
        The GLCMs, where m is the dimension of each GLCM and p is the number of
        GLCMs in the array. Features are computed for each of the p arrays
        provided.

    features : string or list of strings
        Listing the features to compute. If omitted, the default is None, which
        means to compute all features.

        Features available:
        - "autoCorrelation": Autocorrelation [2, 4]
        - "clusterShade": Cluster Shade [2, 4]
        - "clusterProminence": Cluster Prominence [2, 4]
        - "contrast": Contrast (a.k.a. Inertia) [1, 2, 4]
        - "correlation": Correlation [1, 2, 3, 4]
        - "differenceAverage": Difference average ($\mu_{x-y}$ in [4], analog
              to `Sum average`)
        - "differenceEntropy": Difference entropy [1, 4]
        - "differenceVariance": Difference variance [1, 4]
        - "dissimilarity": Dissimilarity: [2, 3, 4]
        - "energy": Energy (a.k.a. Angular Second Moment or
              Uniformity) [1, 2, 4]
        - "entropy": Entropy [1, 2, 3, 4]
        - "homogeneity": Homogeneity: (a.k.a. Inverse Difference
              Moment) [1, 2, 3, 4]
        - "informationMeasureOfCorrelation1": Information measure of
              correlation 1 [1, 4]
        - "informationMeasureOfCorrelation2": Information measure of
              correlation 2 [1, 4]
        - "inverseDifference": Inverse difference: [3, 4]
        - "maximalCorrelationCoefficient": Maximum correlation
              coefficient [1, 4]
        - "maximumProbability": Maximum probability [2, 4]
        - "sumAverage": Sum average [1, 4]
        - "sumEntropy": Sum entropy [1, 4]
        - "sumOfSquaresVariance": Sum of squares: Variance [1, 4]
        - "sumVariance": Sum variance [1, 4]

    Examples
    --------
    >>> import GLCMFeaturesInvariant as GLCM
    >>> import numpy as np
    >>> import skimage.data
    >>> import skimage.feature
    >>>
    >>> # First create a GLCM from a 2d image
    >>> image = skimage.data.camera()
    >>> glcm = skimage.feature.greycomatrix(image,
    ...                                     distances=[1],
    ...                                     angles=[0,
    ...                                             np.pi / 4,
    ...                                             np.pi / 2,
    ...                                             3 * np.pi / 4],
    ...                                     levels=256,
    ...                                     symmetric=False,
    ...                                     normed=False)
    >>> # Sum the GLCMs of different directions to create a direction invariant
    >>> # GLCM
    >>> glcm = np.sum(glcm, axis=-1)
    >>>
    >>> # Compute the invariant Haralick features
    >>> features = GLCM.GLCMFeaturesInvariant(glcm)
    >>> round(features["energy"][0], 3)
    106.711
    >>> round(features["entropy"][0], 3)
    -3.168
    >>> round(features["maximalCorrelationCoefficient"][0], 3)
    0.976
    >>> # Compute energy and entropy only
    >>> features = GLCM.GLCMFeaturesInvariant(glcm, props=["energy", "entropy"])
    >>> round(features["energy"][0], 3)
    106.711
    >>> round(features["entropy"][0], 3)
    -3.168
    >>> round(features["maximalCorrelationCoefficient"][0], 3)
    Traceback (most recent call last):
      ...
    KeyError: 'maximalCorrelationCoefficient'
    >>>
    >>> # Compute all features again:
    >>> features = GLCM.GLCMFeaturesInvariant(glcm)
    >>> # The same features computed using the Matlab code:
    >>> matlab_values = {"autoCorrelation": [0.338765646073735],
    ...                "clusterProminence": [0.183058752615987],
    ...                     "clusterShade": [-0.089896173534216],
    ...                         "contrast": [0.003865580547256],
    ...                      "correlation": [0.976658940105098],
    ...                "differenceAverage": [0.033055162852963],
    ...                "differenceEntropy": [-2.725362168091100],
    ...               "differenceVariance": [0.003015921426747],
    ...                    "dissimilarity": [0.029148912852963],
    ...                           "energy": [1.067107663954530e+02],
    ...                          "entropy": [-3.168061715500240],
    ...                      "homogeneity": [0.996421699584935],
    ... "informationMeasureOfCorrelation1": [3.950478841914947],
    ... "informationMeasureOfCorrelation2": [0.992517789637628],
    ...                "inverseDifference": [0.973992083790672],
    ...    "maximalCorrelationCoefficient": [0.976032758107448],
    ...               "maximumProbability": [6.184983271258128e+02],
    ...                       "sumAverage": [0.507861749022182],
    ...                       "sumEntropy": [0.060784521057008],
    ...             "sumOfSquaresVariance": [0.082839991976549],
    ...                      "sumVariance": [0.292844848248440]}
    >>>
    >>> # Make sure they have the same keys:
    >>> set(matlab_values.keys()).difference(set(features.keys()))
    set()
    >>> # Make sure the values are the same:
    >>> for key in matlab_values.keys():
    ...     # print(abs(features[key][0] - matlab_values[key][0]))
    ...     assert(abs(features[key][0] - matlab_values[key][0]) < 5e-12)

    References
    ----------
    1. Haralick R. M., Shanmugam K., and Dinstein I. Textural Features of Image
       Classification. IEEE Transactions on Systems, Man and Cybernetics, 3(6),
       1973.
    2. Soh L. and Tsatsoulis C., Texture Analysis of SAR Sea Ice Imagery
       Using Gray Level Co-Occurrence Matrices. IEEE Transactions on Geoscience
       and Remote Sensing, 37(2), 1999.
    3. Clausi D. A., An analysis of co-occurrence texture statistics as a
       function of grey level quantization, Can. J. Remote Sensing, 28(1),
       45-62, 2002.
    4. Löfstedt T, Brynolfsson P, Asklund T, Nyholm T, Garpebring A.
       Gray-level invariant Haralick texture features. PLOS ONE, 14(2), 2019.
    """
    # Handle input arguments
    if not isinstance(glcms, np.ndarray):
        glcms = np.asarray(glcms)
    if glcms.ndim == 2:
        glcms = glcms[..., np.newaxis]
    if glcms.ndim != 3 or glcms.shape[0] <= 1 or glcms.shape[1] <= 1:
        raise ValueError("The GLCMs should be 2-D or 3-D matrices.")
    if glcms.shape[0] != glcms.shape[1]:
        raise ValueError("The GLCMs should be square matrices.")

    if props is not None:
        if isinstance(props, str):
            if props.tolower() == "all":
                props = None
            else:
                props = [props]
        elif not isinstance(props, (list, tuple)):
            props = [props]

    homogeneity_constant = max(0.0, float(homogeneity_constant))
    inverse_difference_constant = max(0.0, float(inverse_difference_constant))

    # Size of GLCMs
    num_gray_levels = glcms.shape[0]
    num_glcms = glcms.shape[2]

    # Differentials
    dA = 1.0 / (num_gray_levels**2.0)
    dL = 1.0 / num_gray_levels
    dXplusY = 1.0 / (2.0 * num_gray_levels - 1.0)
    dXminusY = 1.0 / num_gray_levels
    dkdiag = 1.0 / num_gray_levels

    # Normalize the GLCMs
    glcms = glcms.astype(np.float64)
    for i in range(glcms.shape[2]):
        glcms[:, :, i] /= np.sum(np.sum(glcms[:, :, i])) * dA

    glcm_mean = np.zeros((num_glcms,))
    uX = np.zeros((num_glcms,))
    uY = np.zeros((num_glcms,))
    sX = np.zeros((num_glcms,))
    sY = np.zeros((num_glcms,))

    # pX, pY, pXplusY, and pXminusY
    if (props is None
            or strcmp("informationMeasureOfCorrelation1", props)
            or strcmp("informationMeasureOfCorrelation2", props)
            or strcmp("maximalCorrelationCoefficient", props)):
        pX = np.zeros((num_gray_levels, num_glcms))  # Ng x #glcms[1]
        pY = np.zeros((num_gray_levels, num_glcms))  # Ng x #glcms[1]
    if (props is None
            or strcmp("sumAverage", props)
            or strcmp("sumVariance", props)
            or strcmp("sumEntropy", props)):
        pXplusY = np.zeros(((num_gray_levels * 2 - 1), num_glcms))  # [1]
    if (props is None
            or strcmp("differenceEntropy", props)
            or strcmp("differenceVariance", props)):
        pXminusY = np.zeros((num_gray_levels, num_glcms))  # [1]

    # HXY1 HXY2 HX HY
    if (props is None
            or strcmp("informationMeasureOfCorrelation1", props)):
        HXY1 = np.zeros((num_glcms,))
        HX = np.zeros((num_glcms,))
        HY = np.zeros((num_glcms,))
    if (props is None
            or strcmp("informationMeasureOfCorrelation2", props)):
        HXY2 = np.zeros((num_glcms,))

    # Create indices for vectorised code:
    sub = np.arange(0, num_gray_levels * num_gray_levels)
    I, J = np.unravel_index(sub, (num_gray_levels, num_gray_levels))
    nI = (I + 1) / num_gray_levels
    nJ = (J + 1) / num_gray_levels

    if (props is None
            or strcmp("sumAverage", props)
            or strcmp("sumVariance", props)
            or strcmp("sumEntropy", props)):
        sum_lin_ind = [None] * (2 * num_gray_levels)
        for i in range(1, 2 * num_gray_levels):
            diagonal = i - num_gray_levels
            d = np.ones((num_gray_levels - abs(diagonal),))

            diag_ = np.diag(d, k=diagonal)
            diag_ud_ = np.flipud(diag_)
            # sum_lin_ind[i] = np.ravel_multi_index(np.nonzero(diag_ud_),
            #                                       (num_gray_levels,
            #                                        num_gray_levels))
            sum_lin_ind[i] = np.nonzero(diag_ud_)
            # glcm[(*np.nonzero(diag_ud_), 0)]

    if (props is None
            or strcmp("differenceAverage", props)
            or strcmp("differenceVariance", props)
            or strcmp("differenceEntropy", props)):

        diff_lin_ind = [None] * num_gray_levels
        for i in range(num_gray_levels):
            diagonal = i
            d = np.ones((num_gray_levels - diagonal,))
            if diagonal == 0:
                D = np.diag(d, k=diagonal)
                diff_lin_ind[i] = np.nonzero(D)
            else:
                Dp = np.diag(d, k=diagonal)
                Dn = np.diag(d, k=-diagonal)
                diff_lin_ind[i] = np.nonzero(Dp + Dn)

    sum_indices = np.arange(2, 2 * num_gray_levels + 1)

    out = {}

    # Loop over all GLCMs
    for k in range(num_glcms):
        current_glcm = glcms[:, :, k]
        glcm_mean[k] = np.mean(current_glcm)

        # For symmetric GLCMs, uX = uY
        uX[k] = np.sum(np.multiply(nI, current_glcm.ravel()[sub])) * dA
        uY[k] = np.sum(np.multiply(nJ, current_glcm.ravel()[sub])) * dA
        sX[k] = np.sum(np.multiply((nI - uX[k])**2.0,
                                   current_glcm.ravel()[sub])) * dA
        sY[k] = np.sum(np.multiply((nJ - uY[k])**2.0,
                                   current_glcm.ravel()[sub])) * dA

        if (props is None
                or strcmp("sumAverage", props)
                or strcmp("sumVariance", props)
                or strcmp("sumEntropy", props)):
            for i in sum_indices:
                pXplusY[i - 2, k] \
                    = np.sum(current_glcm[sum_lin_ind[i - 1]]) * dkdiag

        if (props is None
                or strcmp("differenceAverage", props)
                or strcmp("differenceVariance", props)
                or strcmp("differenceEntropy", props)):
            idx2 = np.arange(num_gray_levels)
            for i in idx2:
                pXminusY[i, k] = np.sum(current_glcm[diff_lin_ind[i]]) * dkdiag

        if (props is None
                or strcmp("informationMeasureOfCorrelation1", props)
                or strcmp("informationMeasureOfCorrelation2", props)
                or strcmp("maximalCorrelationCoefficient", props)):
            pX[:, k] = np.sum(current_glcm, axis=1) * dL
            pY[:, k] = np.sum(current_glcm, axis=0) * dL
        if (props is None
                or strcmp("informationMeasureOfCorrelation1", props)):
            ind_non_zero = pX[:, k] > MACHINE_EPSILON
            HX[k] = -np.sum(np.multiply(pX[ind_non_zero, k],
                                        np.log(pX[ind_non_zero, k]))) * dL
            ind_non_zero = pY[:, k] > MACHINE_EPSILON
            HY[k] = -np.sum(np.multiply(pY[ind_non_zero, k],
                                        np.log(pY[ind_non_zero, k]))) * dL
            pXpY = np.multiply(pX[I, k], pY[J, k])
            ind_non_zero = pXpY > MACHINE_EPSILON
            HXY1[k] = -np.sum(
                np.multiply(current_glcm.ravel()[sub][ind_non_zero],
                            np.log(pXpY[ind_non_zero]))) * dA
        if (props is None
                or strcmp("informationMeasureOfCorrelation2", props)):
            pXpY = np.multiply(pX[I, k], pY[J, k])
            ind_non_zero = pXpY > MACHINE_EPSILON
            HXY2[k] = -np.sum(np.multiply(pXpY,
                                          np.log(pXpY[ind_non_zero]))) * dA

        # Haralick features:
        # -----------------
        if (props is None
                or strcmp("energy", props)):
            if "energy" not in out:
                out["energy"] = [None] * num_glcms
            out["energy"][k] = np.sum(current_glcm.ravel()[sub]**2.0) * dA
        if (props is None
                or strcmp("contrast", props)):
            if "contrast" not in out:
                out["contrast"] = [None] * num_glcms
            out["contrast"][k] = np.sum(
                    np.multiply((nI - nJ)**2.0,
                                current_glcm.ravel()[sub])) * dA

        if (props is None
                or strcmp("autoCorrelation", props)
                or strcmp("correlation", props)):
            auto_correlation = np.sum(
                    np.multiply(np.multiply(nI, nJ),
                                current_glcm.ravel()[sub])) * dA
            if (props is None
                    or strcmp("autoCorrelation", props)):
                if "autoCorrelation" not in out:
                    out["autoCorrelation"] = [None] * num_glcms
                out["autoCorrelation"][k] = auto_correlation

        if (props is None
                or strcmp("correlation", props)):
            if "correlation" not in out:
                out["correlation"] = [None] * num_glcms
            if sX[k] < MACHINE_EPSILON or sY[k] < MACHINE_EPSILON:
                out["correlation"][k] \
                    = min(max((auto_correlation - uX[k] * uY[k]), -1.0), 1.0)
            else:
                out["correlation"][k] \
                    = (auto_correlation - uX[k] * uY[k]) \
                    / np.sqrt(sX[k] * sY[k])
        if (props is None
                or strcmp("sumOfSquaresVariance", props)):
            if "sumOfSquaresVariance" not in out:
                out["sumOfSquaresVariance"] = [None] * num_glcms
            out["sumOfSquaresVariance"][k] \
                = np.sum(np.multiply(current_glcm.ravel()[sub],
                                     (nI - uX[k])**2.0)) * dA
        if (props is None
                or strcmp("homogeneity", props)):
            if "homogeneity" not in out:
                out["homogeneity"] = [None] * num_glcms
            out["homogeneity"][k] \
                = np.sum(np.divide(
                    current_glcm.ravel()[sub],
                    1.0 + homogeneity_constant * (nI - nJ)**2.0)) * dA
        if (props is None
                or strcmp("sumAverage", props)
                or strcmp("sumVariance", props)):
            sum_average = np.sum(np.multiply(
                (2.0 * (sum_indices - 1.0)) / (2.0 * num_gray_levels - 1.0),
                pXplusY[sum_indices - 2, k])) * dXplusY
            if (props is None
                    or strcmp("sumAverage", props)):
                if "sumAverage" not in out:
                    out["sumAverage"] = [None] * num_glcms
                out["sumAverage"][k] = sum_average
        if (props is None
                or strcmp("sumVariance", props)):
            if "sumVariance" not in out:
                out["sumVariance"] = [None] * num_glcms
            out["sumVariance"][k] = np.sum(np.multiply(
                (((2.0 * (sum_indices - 1.0)) / (2.0 * num_gray_levels - 1.0))
                    - sum_average)**2.0,
                pXplusY[sum_indices - 2, k])) * dXplusY
        if (props is None
                or strcmp("sumEntropy", props)):
            if "sumEntropy" not in out:
                out["sumEntropy"] = [None] * num_glcms
            pXplusY_sim_indices_k = pXplusY[sum_indices - 2, k]
            ind_non_zero = pXplusY_sim_indices_k > MACHINE_EPSILON
            out["sumEntropy"][k] = -np.sum(  # Differential entropy
                np.multiply(pXplusY_sim_indices_k[ind_non_zero],
                            np.log(pXplusY_sim_indices_k[ind_non_zero]))
                ) * dXplusY
        if (props is None
                or strcmp("entropy", props)
                or strcmp("informationMeasureOfCorrelation1", props)
                or strcmp("informationMeasureOfCorrelation2", props)):
            current_glcm_sub = current_glcm.ravel()[sub]
            ind_non_zero = current_glcm_sub > MACHINE_EPSILON
            entropy = -np.sum(  # Differential entropy
                np.multiply(current_glcm_sub[ind_non_zero],
                            np.log(current_glcm_sub[ind_non_zero]))) * dA
            if (props is None
                    or strcmp("entropy", props)):
                if "entropy" not in out:
                    out["entropy"] = [None] * num_glcms
                out["entropy"][k] = entropy

        if (props is None
                or strcmp("differenceAverage", props)
                or strcmp("differenceVariance", props)):
            difference_average = np.sum(np.multiply(
                (idx2 + 1) / num_gray_levels,
                pXminusY[idx2, k])) * dXminusY
            if (props is None
                    or strcmp("differenceAverage", props)):
                if "differenceAverage" not in out:
                    out["differenceAverage"] = [None] * num_glcms
                out["differenceAverage"][k] = difference_average
        if (props is None
                or strcmp("differenceVariance", props)):
            if "differenceVariance" not in out:
                out["differenceVariance"] = [None] * num_glcms
            out["differenceVariance"][k] = np.sum(np.multiply(
                    (((idx2 + 1) / num_gray_levels) - difference_average)**2.0,
                    pXminusY[idx2, k])) * dXminusY
        if (props is None
                or strcmp("differenceEntropy", props)):
            if "differenceEntropy" not in out:
                out["differenceEntropy"] = [None] * num_glcms
            pXminusY_idx2_k = pXminusY[idx2, k]
            ind_non_zero = pXminusY_idx2_k > MACHINE_EPSILON
            # Differential entropy
            out["differenceEntropy"][k] = -np.sum(np.multiply(
                pXminusY_idx2_k[ind_non_zero],
                np.log(pXminusY_idx2_k[ind_non_zero]))) * dXminusY

        if (props is None
                or strcmp("informationMeasureOfCorrelation1", props)):
            info_measure_1 = (entropy - HXY1[k]) / max(HX[k], HY[k])
            if "informationMeasureOfCorrelation1" not in out:
                out["informationMeasureOfCorrelation1"] = [None] * num_glcms
            out["informationMeasureOfCorrelation1"][k] = info_measure_1
        if (props is None
                or strcmp("informationMeasureOfCorrelation2", props)):
            info_measure_2 = np.sqrt(1.0 - np.exp(-2.0 * (HXY2[k] - entropy)))
            if "informationMeasureOfCorrelation2" not in out:
                out["informationMeasureOfCorrelation2"] = [None] * num_glcms
            out["informationMeasureOfCorrelation2"][k] = info_measure_2
        if (props is None
                or strcmp("maximalCorrelationCoefficient", props)):
            # Correct by eps if the matrix has columns or rows that sums to zero
            P = current_glcm
            pX_ = pX[:, k]
            if np.any(pX_ < MACHINE_EPSILON):
                pX_ = pX_ + MACHINE_EPSILON  # TODO: Only increase the small?
                pX_ = pX_ / (np.sum(pX_) * dL)  # Renormalise
            pY_ = pY[:, k]
            if np.any(pY_ < MACHINE_EPSILON):
                pY_ = pY_ + MACHINE_EPSILON
                pY_ = pY_ / (np.sum(pY_) * dL)

            # Compute the Markov matrix
            Q = np.zeros(P.shape)
            # for i = 1:nGrayLevels
            for i in range(num_gray_levels):
                Pi = P[i, :]
                pXi = pX_[i]
                # for j = 1:nGrayLevels
                for j in range(num_gray_levels):
                    Pj = P[j, :]
                    d = pXi * pY_
                    if np.any(d < MACHINE_EPSILON):
                        raise RuntimeError("Division by zero in the "
                                           "maximalCorrelationCoefficient!")
                    Q[i, j] = np.sum(np.divide(np.multiply(Pi, Pj), d)) * dA

            # Compute the second largest eigenvalue of Q
            if np.any(np.isinf(Q)):
                e2 = np.nan
            else:
                try:
                    E = scipy.sparse.linalg.eigs(Q,
                                                 k=2,
                                                 return_eigenvectors=False)
                except Exception:
                    try:
                        E = scipy.linalg.eigvals(Q)
                    except Exception:
                        raise RuntimeError("Could not compute the "
                                           "maximalCorrelationCoefficient!")

                # There may be a near-zero imaginary component here
                E = np.real(E)
                if E.shape[0] < 2:  # Did we find at least two eigenvalues?
                    raise RuntimeError("Could not compute the "
                                       "maximalCorrelationCoefficient!")
                # Find the second largest eigenvalue. Note: The order may be
                # changed if there's a non-zero imaginary component, hence the
                # use of min here.
                if E[0] >= E[-1]:
                    e2 = min(E[0], E[1])
                else:
                    e2 = min(E[-2], E[-1])

            if "maximalCorrelationCoefficient" not in out:
                out["maximalCorrelationCoefficient"] = [None] * num_glcms
            out["maximalCorrelationCoefficient"][k] = e2

        if (props is None
                or strcmp("dissimilarity", props)):
            dissimilarity = np.sum(np.multiply(np.abs(nI - nJ),
                                               current_glcm.ravel()[sub])) * dA
            if "dissimilarity" not in out:
                out["dissimilarity"] = [None] * num_glcms
            out["dissimilarity"][k] = dissimilarity
        if (props is None
                or strcmp("clusterShade", props)):
            if "clusterShade" not in out:
                out["clusterShade"] = [None] * num_glcms
            out["clusterShade"][k] = np.sum(
                np.multiply((nI + nJ - uX[k] - uY[k])**3.0,
                            current_glcm.ravel()[sub])) * dA
        if (props is None
                or strcmp("clusterProminence", props)):
            if "clusterProminence" not in out:
                out["clusterProminence"] = [None] * num_glcms
            out["clusterProminence"][k] = np.sum(
                np.multiply((nI + nJ - uX[k] - uY[k])**4.0,
                            current_glcm.ravel()[sub])) * dA
        if (props is None
                or strcmp("maximumProbability", props)):
            if "maximumProbability" not in out:
                out["maximumProbability"] = [None] * num_glcms
            out["maximumProbability"][k] = np.max(current_glcm.ravel()[sub])
        if (props is None
                or strcmp("inverseDifference", props)):
            if "inverseDifference" not in out:
                out["inverseDifference"] = [None] * num_glcms
            out["inverseDifference"][k] = np.sum(
                np.divide(current_glcm.ravel()[sub],
                          1.0 + inverse_difference_constant * np.abs(nI - nJ)
                          )) * dA
    return out


def strcmp(string, strings):
    """Case-insensitive string-in-list function.

    Returns true if the given string is in the list of strings, with
    case-insensitive comparison. Also returns True if "all" is in the list of
    strings.
    """
    if isinstance(strings, str):
        strings = [strings]
    return (string.lower() in (str_.lower() for str_ in strings)) \
        or ("all" in (str_.lower() for str_ in strings))


if __name__ == "__main__":
    try:
        import doctest
        doctest.testmod()
    except ModuleNotFoundError:
        raise RuntimeError("Need the 'doctest' package to run tests.")
