#pragma once

#include <armadillo>

/**
 * @brief      Calculates the pairwise distances.
 *
 * @param[in]  points  NxD matrix of N points in R^D space
 *
 * @return     NxN matrix of pairwise Euclidean distances between each of the points.
 */
arma::mat calculate_pairwise_distances(const arma::mat& points);

/**
 * @brief      Given pairwise distances the function calculates
 *             the conditional similarity of the points with constant sigma
 *
 * @param[in]  pairwise_distances  NxN matrix of
 *             the pairwise distances between the points
 *
 * @param[in]  sigma   The Gaussian sigma
 *
 * @return     NxN matrix of the conditional similarity
 */
arma::mat calcualate_gaussian_condition_similarity_constant_sigma(
    const arma::mat& pairwise_distances,
    const double sigma);


/**
 * @brief      Calculate similaraty of the map points using the t-Student distribution
 *
 * @param[in]  pairwise_distances  NxN matrix of
 *             the pairwise distances between the points
 *
 * @return     NxN matrix of the conditional similarity
 */
arma::mat calcualate_tstudent_condition_similarity(
    const arma::mat& pairwise_distances);
