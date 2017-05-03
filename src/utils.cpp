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

arma::mat calcualate_gaussian_condition_similarity(
        const arma::mat& pairwise_distances,
        const arma::vec& sigma) {

    arma::vec sigma2 = 2 * sigma % sigma;
    arma::mat similarites(pairwise_distances.n_rows, pairwise_distances.n_cols);

    for (size_t n_point = 0; n_point < pairwise_distances.n_rows; ++n_point) {
        similarites.row(n_point) = arma::exp(-pairwise_distances.row(n_point) / sigma2(n_point));
    }

    arma::rowvec norm = arma::sum(similarites, 0);

    auto number_of_points = similarites.n_rows;

    for (size_t n_point = 0; n_point < pairwise_distances.n_rows; ++n_point) {
        similarites.row(n_point) /= norm;
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

double calculate_entropy(arma::rowvec distances, const double beta) {
    arma::rowvec similarites = arma::exp(-distances * beta);
    double norm = arma::sum(similarites);

    double entropy = std::log(norm) + beta * arma::sum(distances % similarites) / norm;
    return entropy;
}

arma::vec calculate_optimal_sigma(const arma::mat image_points,
        const double perplexity, const double eps) {
    auto number_of_points = image_points.n_rows;

    arma::vec sigma(number_of_points);
    const auto pairwise_distances = calculate_pairwise_distances(image_points);

    for (size_t n_point = 0; n_point < number_of_points; ++n_point) {
        auto sigma_min = -arma::datum::inf;
        auto sigma_max = +arma::datum::inf;
        size_t iteration = 0;
        arma::rowvec distances = pairwise_distances.row(n_point);
        distances.shed_col(n_point);

        double current_sigma = 1;
        double current_entropy = calculate_entropy(distances, current_sigma);
        double target_entropy = std::log(perplexity);

        double diff = current_entropy - target_entropy;

        while (std::abs(diff) > eps && iteration < 50) {
            if (diff > 0) {
                sigma_min = current_sigma;
                if (sigma_max == arma::datum::inf || sigma_max == -arma::datum::inf) {
                    current_sigma *= 2;
                } else {
                    current_sigma = (current_sigma + sigma_max) / 2;
                }
            } else {
                sigma_max = current_sigma;
                if (sigma_min == arma::datum::inf || sigma_min == -arma::datum::inf) {
                    current_sigma /= 2;
                } else {
                    current_sigma = (current_sigma + sigma_min) / 2;
                }
            }

            current_entropy = calculate_entropy(distances, current_sigma);
            diff = current_entropy - target_entropy;
            iteration += 1;

        }
        sigma(n_point) = current_sigma;
    }

    sigma = 1 / arma::sqrt(sigma);

    return sigma;
}
