#include "utils.hpp"
#include <cmath>

arma::mat calculate_pairwise_distances(const arma::mat& points) {
    auto number_of_points = points.n_rows;

    arma::mat pairwise_distances(
        number_of_points,
        number_of_points,
        arma::fill::zeros);

    for (size_t n_row = 0; n_row < number_of_points; ++n_row) {
        for (size_t n_col = 0; n_col < n_row; ++n_col) {
            auto distance = arma::sum(arma::square(points.row(n_col) - points.row(n_row)));

            pairwise_distances(n_row, n_col) = \
                pairwise_distances(n_col, n_row) = \
                    distance;
        }
    }

    return pairwise_distances;
}

arma::mat calcualate_gaussian_condition_similarity_constant_sigma(
        const arma::mat& pairwise_distances,
        const double sigma) {

    double sigma2 = 2 * sigma * sigma;
    arma::mat similarites =  arma::exp(-pairwise_distances / sigma2);

    arma::rowvec norm = arma::sum(similarites, 0);

    auto number_of_points = similarites.n_rows;

    for (size_t n_row = 0; n_row < pairwise_distances.n_rows; ++n_row) {
        similarites.row(n_row) /= norm;
    }

    return (similarites + similarites.t()) / (2 * number_of_points);
}


arma::mat calcualate_tstudent_condition_similarity(
        const arma::mat& pairwise_distances) {

    arma::mat similarites = 1 / (1 + pairwise_distances);

    arma::rowvec norm = arma::sum(similarites, 0);

    auto number_of_points = similarites.n_rows;

    for (size_t n_row = 0; n_row < pairwise_distances.n_rows; ++n_row) {
        similarites.row(n_row) /= norm;
    }

    return (similarites + similarites.t()) / (2 * number_of_points);
}
