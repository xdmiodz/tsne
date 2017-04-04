#include <cmath>
#include "optimization.hpp"

arma::mat calculate_tsne_gradient(
        const arma::mat& image_similarities,
        const arma::mat& map_points) {

    const size_t n_points = map_points.n_rows;
    const size_t n_dimensions = map_points.n_cols;

    const auto map_pairwise_distances = calculate_pairwise_distances(map_points);
    const auto map_similarities = calcualate_tstudent_condition_similarity(map_pairwise_distances);

    arma::mat grad_coeffs = 4 * (image_similarities - map_similarities) \
        % arma::sqrt(map_pairwise_distances) / (1 + map_pairwise_distances);

    arma::mat grad(n_points, n_dimensions, arma::fill::zeros);

    for (size_t point_idx = 0; point_idx < n_points; ++point_idx) {
        for (size_t neighbour_idx = 0; neighbour_idx < n_points; ++neighbour_idx) {
            if (neighbour_idx == point_idx) {
                continue;
            }
            arma::rowvec diff = map_points.row(point_idx) - map_points.row(neighbour_idx);

            grad.row(point_idx) += grad_coeffs(point_idx, neighbour_idx) * diff;
        }
    }

    return grad;
}

double calculate_loss(
        const arma::mat& image_points,
        const arma::mat& map_points
    ) {

    const auto image_pairwise_distances = calculate_pairwise_distances(image_points);
    const auto image_similarities = \
        calcualate_gaussian_condition_similarity_constant_sigma(image_pairwise_distances, 10);

    const auto map_pairwise_distances = calculate_pairwise_distances(map_points);
    const auto map_similarities = \
        calcualate_tstudent_condition_similarity(map_pairwise_distances);

    return arma::sum(arma::sum(image_similarities % arma::log(image_similarities / map_similarities)));
}

arma::mat run_tsne_optimization(
        const arma::mat& image_points,
        arma::mat map_points,
        const size_t max_iterations,
        const double eps,
        const double learning_rate,
        const double momentum
    ) {

    const auto image_pairwise_distances = calculate_pairwise_distances(image_points);
    const auto image_similarities = \
        calcualate_gaussian_condition_similarity_constant_sigma(image_pairwise_distances, 10);


    const auto initial_grad = calculate_tsne_gradient(image_similarities, map_points);
    auto grad = initial_grad;

    auto is_gradient_close = [eps, &initial_grad] (const arma::mat& grad) {
        return arma::norm(grad) <= std::sqrt(eps) * arma::norm(initial_grad);
    };

    size_t iteration_number = 0;
    while (iteration_number < max_iterations && !is_gradient_close(grad)) {
        map_points -= learning_rate * grad;
        grad = calculate_tsne_gradient(image_similarities, map_points);
    }

    return map_points;
}
