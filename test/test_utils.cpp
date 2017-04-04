#define BOOST_TEST_MODULE test_utils
#include <boost/test/included/unit_test.hpp>

#include "src/utils.hpp"

arma::mat make_test_points() {
    arma::mat points(4, 2);

    points(0, 0) = 0; points(1, 1) = 0;
    points(1, 0) = 3; points(1, 1) = 4;
    points(2, 0) = 2; points(2, 1) = 5;
    points(3, 0) = 2; points(3, 1) = 5;

    return points;
}

BOOST_AUTO_TEST_CASE(test_sanity_check_pairwise_distances) {
    auto points = make_test_points();

    auto pairwise_distances = calculate_pairwise_distances(points);

    BOOST_CHECK_EQUAL(pairwise_distances.n_cols, pairwise_distances.n_rows);
    BOOST_CHECK_EQUAL(pairwise_distances.n_cols, 4);

    BOOST_CHECK_EQUAL(pairwise_distances(0, 0), 0);
    BOOST_CHECK_EQUAL(pairwise_distances(1, 1), 0);
    BOOST_CHECK_EQUAL(pairwise_distances(2, 2), 0);
    BOOST_CHECK_EQUAL(pairwise_distances(3, 3), 0);

    BOOST_CHECK(arma::approx_equal(
        pairwise_distances, pairwise_distances.t(), "absdiff", 1e-6));

    BOOST_CHECK_CLOSE(pairwise_distances(0, 1), 25, 1e-6);
    BOOST_CHECK_CLOSE(pairwise_distances(0, 2), 29, 1e-6);
    BOOST_CHECK_CLOSE(pairwise_distances(0, 3), 29, 1e-6);

    BOOST_CHECK_CLOSE(pairwise_distances(1, 2), 2, 1e-6);
    BOOST_CHECK_CLOSE(pairwise_distances(1, 3), 2, 1e-6);

    BOOST_CHECK_CLOSE(pairwise_distances(2, 3), 0, 1e-6);
}


BOOST_AUTO_TEST_CASE(test_calcualate_gaussian_condition_similarity_constant_sigma) {
    auto points = make_test_points();

    double sigma = 2;

    auto pairwise_distances = calculate_pairwise_distances(points);

    auto condition_similarity = \
        calcualate_gaussian_condition_similarity_constant_sigma(pairwise_distances, sigma);

    arma::mat reference(4, 4);
    reference << 0.22784542 << 0.00711652 << 0.00422332 << 0.00422332 << arma::endr
              << 0.00711652 << 0.09609698 << 0.07212055 << 0.07212055 << arma::endr
              << 0.00422332 << 0.07212055 << 0.08911227 << 0.08911227 << arma::endr
              << 0.00422332 << 0.07212055 << 0.08911227 << 0.08911227 << arma::endr;

    BOOST_CHECK(arma::approx_equal(condition_similarity, reference, "absdiff", 1e-6));
}

BOOST_AUTO_TEST_CASE(test_calcualate_tstudent_condition_similarity) {
    auto points = make_test_points();

    auto pairwise_distances = calculate_pairwise_distances(points);

    auto condition_similarity = \
        calcualate_tstudent_condition_similarity(pairwise_distances);

    arma::mat reference(4, 4);

    reference << 0.2262181  << 0.0071699  << 0.00553087 << 0.00553087 << arma::endr
              << 0.0071699  << 0.14661654 << 0.04204172 << 0.04204172 << arma::endr
              << 0.00553087 << 0.04204172 << 0.1056338  << 0.1056338 << arma::endr
              << 0.00553087 << 0.04204172 << 0.1056338  << 0.1056338 << arma::endr;

    BOOST_CHECK(arma::approx_equal(condition_similarity, reference, "absdiff", 1e-6));
}

BOOST_AUTO_TEST_CASE(test_perf_check_pairwise_distances) {
    arma::mat points(1000, 512, arma::fill::randu);
    auto pairwise_distances = calculate_pairwise_distances(points);
    BOOST_CHECK(arma::approx_equal(pairwise_distances, pairwise_distances, "absdiff", 1e-6));
}

