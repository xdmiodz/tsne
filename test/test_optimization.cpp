#define BOOST_TEST_MODULE test_optimization
#include <boost/test/included/unit_test.hpp>

#include "src/utils.hpp"
#include "src/optimization.hpp"

arma::mat make_test_image_points() {
    arma::mat points(4, 6);
    points << 2 << -1 << -7 << -2 << -2 <<  5 << arma::endr
           << 6 <<  3 << -2 <<  4 << -5 <<  6 << arma::endr
           << 8 << -6 <<  5 <<  8 << -6 << -8 << arma::endr
           << 4 <<  6 <<  2 <<  0 <<  0 <<  0 << arma::endr;
    return points;
}

arma::mat make_test_map_points() {
    arma::mat points(4, 6);
    points <<  6 <<  7 << arma::endr
           <<  7 << -2 << arma::endr
           << -7 << -2 << arma::endr
           <<  8 << -6 << arma::endr;
    return points;
}

BOOST_AUTO_TEST_CASE(test_calculate_tsne_gradient) {
    auto image_points = make_test_image_points();

    auto map_points = make_test_map_points();

    const auto image_pairwise_distances = calculate_pairwise_distances(image_points);
    const auto image_similarities = \
        calcualate_gaussian_condition_similarity_constant_sigma(image_pairwise_distances, 10);

    auto grad = calculate_tsne_gradient(
        image_similarities,
        map_points
    );

    arma::mat reference(4, 2);

    reference << -0.0196 << 0.4692  << arma::endr
              <<  0.0736 << -0.0629 << arma::endr
              << -0.2306 << 0.0005  << arma::endr
              << 0.1766  << -0.4068 << arma::endr;

    BOOST_CHECK(arma::approx_equal(grad, reference, "absdiff", 1e-4));
}

BOOST_AUTO_TEST_CASE(test_run_tsne_optimization) {
    auto image_points = make_test_image_points();
    auto map_points = make_test_map_points();

    auto loss_before = calculate_loss(image_points, map_points);

    auto result_points = run_tsne_optimization(image_points, map_points, 100, 1e-6);

    auto loss_after = calculate_loss(image_points, result_points);

    BOOST_CHECK(result_points.n_rows == map_points.n_rows);
    BOOST_CHECK(result_points.n_cols == map_points.n_cols);
    BOOST_CHECK(loss_after < loss_before);
}
