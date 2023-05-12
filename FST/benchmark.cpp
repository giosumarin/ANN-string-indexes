#include "fst_mod.hpp"
#include "utils.hpp"

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "args: dataset1 [dataset2 ...]" << std::endl;
        return 1;
    }

    std::cout << "dataset,n,error,trie+suffixes bytes,trie bytes,nodes,ns/lookup" << std::endl;
    for (int argi = 1; argi < argc; ++argi) {
        auto data = read_string_file(argv[argi]);
        auto dataset_name = basename(argv[argi]);
        if (!std::is_sorted(data.begin(), data.end())) {
            std::cerr << dataset_name << " is not sorted" << std::endl;
            exit(1);
        }

        #pragma omp parallel for ordered // NOLINT(openmp-use-default-none)
        for (size_t error = 0; error <= 10; error += 2) {
            auto sampled_data = error == 0 ? data : regular_rate_sampling(data, 2 * error);
            auto queries1 = generate_queries(sampled_data);
            auto queries2 = generate_queries(sampled_data);
            auto queries3 = generate_queries(sampled_data);

            fst::Trie surf(sampled_data);
            sampled_data = {};

            auto do_queries = [&](auto &q) { return query_time([&](auto &s) { return surf.exactSearch(s); }, q); };

            #pragma omp ordered
            std::cout << dataset_name
                      << "," << data.size()
                      << "," << error
                      << "," << surf.getMemoryUsage()
                      << "," << surf.getMemoryUsageNoSuffixes()
                      << "," << surf.getNumNodes()
                      << "," << (do_queries(queries1) + do_queries(queries2) + do_queries(queries3)) / 3
                      << std::endl;
        }

        #pragma omp parallel for ordered // NOLINT(openmp-use-default-none)
        for (size_t error = 20; error <= 1000; error += 10) {
            auto sampled_data = error == 0 ? data : regular_rate_sampling(data, 2 * error);
            auto queries1 = generate_queries(sampled_data);
            auto queries2 = generate_queries(sampled_data);
            auto queries3 = generate_queries(sampled_data);

            fst::Trie surf(sampled_data);
            sampled_data = {};

            auto do_queries = [&](auto &q) { return query_time([&](auto &s) { return surf.exactSearch(s); }, q); };

            #pragma omp ordered
            std::cout << dataset_name
                      << "," << data.size()
                      << "," << error
                      << "," << surf.getMemoryUsage()
                      << "," << surf.getMemoryUsageNoSuffixes()
                      << "," << surf.getNumNodes()
                      << "," << (do_queries(queries1) + do_queries(queries2) + do_queries(queries3)) / 3
                      << std::endl;
        }

        #pragma omp parallel for ordered // NOLINT(openmp-use-default-none)
        for (size_t error = 1100; error <= 10000; error += 100) {
            auto sampled_data = error == 0 ? data : regular_rate_sampling(data, 2 * error);
            auto queries1 = generate_queries(sampled_data);
            auto queries2 = generate_queries(sampled_data);
            auto queries3 = generate_queries(sampled_data);

            fst::Trie surf(sampled_data);
            sampled_data = {};

            auto do_queries = [&](auto &q) { return query_time([&](auto &s) { return surf.exactSearch(s); }, q); };

            #pragma omp ordered
            std::cout << dataset_name
                      << "," << data.size()
                      << "," << error
                      << "," << surf.getMemoryUsage()
                      << "," << surf.getMemoryUsageNoSuffixes()
                      << "," << surf.getNumNodes()
                      << "," << (do_queries(queries1) + do_queries(queries2) + do_queries(queries3)) / 3
                      << std::endl;
        }
    }
    return 0;
}
