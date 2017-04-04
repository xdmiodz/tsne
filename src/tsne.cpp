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
    double learning_rate;
    double momentum;
    double eps;

    desc.add_options()
        ("help,h", "Show help")
        ("image-file,i", po::value<std::string>(&image_file), "A file with image points")
        ("map-file,o", po::value<std::string>(&map_file), "A file where map points will be stored")
        ("learning-rate,l", po::value<double>(&learning_rate), "Learning rate value of gradient descent")
        ("eps,e", po::value<double>(&eps), "Maximum realtive error of gradient")
        ("momentum,m", po::value<double>(&momentum), "Momentum of gradient descent")
    ;

    po::variables_map vm;
    po::parsed_options parsed = \
        po::command_line_parser(argc, argv) \
            .options(desc) \
            .allow_unregistered() \
            .run();

    po::store(parsed, vm);
    po::notify(vm);

    if (vm.count("image_file") == 0 || vm.count("help") > 0) {
        std::cout << desc << std::endl;
        return 0;
    }

    return 0;
}
