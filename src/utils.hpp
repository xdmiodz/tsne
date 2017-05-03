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
 * @brief      Given pairwise distances the function calculates
 *             the conditional similarity of the points with sigma assigned
 *             to each of the points
 *
 * @param[in]  pairwise_distances  NxN matrix of
 *             the pairwise distances between the points
 *
 * @param[in]  sigma N vector The gaussian sigma for eaach of the points
 *
 * @return     NxN matrix of the conditional similarity
 */
arma::mat calcualate_gaussian_condition_similarity(
    const arma::mat& pairwise_distances,
    const arma::vec& sigma);


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

/**
 * @brief      Calculates the entropy for given point.
 *
 * @param[in]  distances   Distances to the neighbors for a point, excluding itself
 * @param[in]  sigma       Sigma value
 *
 * @return     The perplexity value
 */
double calculate_entropy(arma::rowvec distances, const double sigma);


/**
 * @brief      Finds optimal sigma foer each of the image points
 *
 * @param[in]  image_points  The image points
 * @param[in]  perplexity    The target perplexity perplexity
 * @param[in]  eps           Maximum error of the calculations
 *
 * @return     Optimal sigmas for eache of the points
 */
arma::vec calculate_optimal_sigma(const arma::mat image_points,
    const double perplexity=30,
    const double eps=1e-5);
