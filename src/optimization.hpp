#pragma once

#include <armadillo>
#include <cmath>
#include "utils.hpp"

/**
 * @brief      Calculates the gradient of the KL logarithm regarding the map points
 *
 * @param[in]  image_similarities  NxN array The similarities of the image points
 * @param[in]  map_points          NxM array The map points
 *
 * @return     NxM array The tsne gradient for each of the map points
 */
arma::mat calculate_tsne_gradient(
        const arma::mat& image_similarities,
        const arma::mat& map_points
    );

/**
 * @brief      Calculates the KL logarithm
 *
 * @param[in]  image_points  NxD array The image points
 * @param[in]  map_points    NxM array The map points
 *
 * @return     double The value of the KL logarithm for the points.
 */
double calculate_loss(
        const arma::mat& image_points,
        const arma::mat& map_points);

/**
 * @brief      Optimiz KL logarithm using the momentum gradient descent
 *
 * @param[in]  image_points        NxD array The image points
 * @param[in]  initial_map_points  NxM The initial map points
 * @param[in]  max_iterations      The maximum number of optimization steps
 * @param[in]  eps                 The relative error of gradient: ||grad_k||^2 <= eps * ||grad_0||^2
 * @param[in]  learning_rate       The learning rate
 * @param[in]  momentum            The momentum coefficient
 *
 * @return     NxM The resulting map points
 */
arma::mat run_tsne_optimization(
        const arma::mat& image_points,
        arma::mat initial_map_points,
        const size_t max_iterations=1000,
        const double eps=1e-6,
        double learning_rate=100
    );
