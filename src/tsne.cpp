#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <armadillo>

#include "utils.hpp"
#include "optimization.hpp"

namespace po = boost::program_options;

int main(int argc, char const *argv[]) {
    po::options_description desc("Run tSNE algorithm");

    std::string image_file;
    std::string map_file;
    double learning_rate = 100;
    double eps = 1e-9;
    size_t map_points_dimension = 2;
    double perplexity = 10;


    desc.add_options()
        ("help,h", "Show help")
        ("image-file,i", po::value<std::string>(&image_file), "A file with image points")
        ("map-file,o", po::value<std::string>(&map_file), "A file where map points will be stored")
        ("dimension,d", po::value<size_t>(&map_points_dimension), "Mapping space dimension")
        ("perplexity,p", po::value<double>(&perplexity), "Perplexity of image distribution")
    ;

    po::variables_map vm;
    po::parsed_options parsed = \
        po::command_line_parser(argc, argv) \
            .options(desc) \
            .allow_unregistered() \
            .run();

    po::store(parsed, vm);
    po::notify(vm);

    if (vm.count("image-file") == 0 || vm.count("help,h") > 0) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::cout << "Loading file " << image_file << std::endl;

    arma::mat image_points;

    if (false == image_points.load(image_file, arma::raw_ascii)) {
        std::cout << "Loading failed, please check the format" << std::endl;
    }

    arma::mat map_points = arma::randu<arma::mat>(image_points.n_rows, map_points_dimension);

    std::cout << "Running tSNE algorithm" << std::endl;

    map_points = run_tsne_optimization(image_points, map_points,
        perplexity);

    if (vm.count("map-file") > 0) {
        map_points.save(map_file, arma::raw_ascii);
        std::cout << "Stored results to " << map_file << std::endl;
    } else {
        std::cout << map_points << std::endl;
    }

    return 0;
}
