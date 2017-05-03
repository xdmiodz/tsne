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

            grad.row(point_idx) += grad_coeffs(point_idx, neighbour_idx) * (map_points.row(point_idx) - map_points.row(neighbour_idx));
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
        calcualate_gaussian_condition_similarity_constant_sigma(image_pairwise_distances, 11);

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
        double learning_rate
    ) {

    const auto image_pairwise_distances = calculate_pairwise_distances(image_points);

    arma::vec sigma =  calculate_optimal_sigma(image_points, 30);

    const auto image_similarities = \
        calcualate_gaussian_condition_similarity(image_pairwise_distances, sigma);

    double momentum = 0.5;

    const auto initial_grad = calculate_tsne_gradient(image_similarities, map_points);
    auto grad = initial_grad;

    auto is_gradient_close = [eps, &initial_grad] (const arma::mat& grad) {
        return arma::norm(grad) <= std::sqrt(eps) * arma::norm(initial_grad);
    };

    size_t iteration_number = 0;

    arma::mat update(map_points.n_rows, map_points.n_cols, arma::fill::zeros);

    while (iteration_number < max_iterations && !is_gradient_close(grad)) {
        update = momentum * update + learning_rate * grad;

        map_points -= update;

        grad = calculate_tsne_gradient(image_similarities, map_points);

        iteration_number++;

        if ((iteration_number % 100) == 0 && (iteration_number > 0)) {
            // learning_rate /= 1.;
            std::cout << "loss: " << calculate_loss(image_points, map_points) << std::endl;
        }

        if (iteration_number > 250) {
            momentum = 0.8;
        }
    }

    return map_points;
}
