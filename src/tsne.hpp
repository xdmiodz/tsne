#pragma once

#include <armadillo>
#include <vector>

class tSNE {
private:
    const arma::mat _image_points;
    arma::mat _image_similarities;
    arma::mat _map_points;

    std::vector<double> _loss_history;

public:
    tSNE(const arma::mat image_points);

};
