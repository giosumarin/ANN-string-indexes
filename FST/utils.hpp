#pragma once

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <chrono>

using timer = std::chrono::high_resolution_clock;

std::string basename(const std::string &path) { return path.substr(path.find_last_of("/\\") + 1); }

std::vector<std::string> read_string_file(const std::string &path, size_t limit = std::numeric_limits<size_t>::max()) {
    auto previous_value = std::ios::sync_with_stdio(false);
    std::vector<std::string> result;
    std::ifstream in(path.c_str());
    if (in.fail()) {
        std::cerr << "Error: " << strerror(errno);
        exit(1);
    }
    std::string str;
    while (std::getline(in, str) && limit-- > 0)
        result.push_back(str);
    std::ios::sync_with_stdio(previous_value);
    return result;
}

template <typename T>
std::vector<T> generate_queries(const std::vector<T> &data) {
    std::vector<T> queries;
    std::sample(data.begin(), data.end(), std::back_inserter(queries),
                std::min<size_t>(data.size(), 10000000), std::mt19937{});
    std::shuffle(queries.begin(), queries.end(), std::mt19937{});
    return queries;
}

template<typename T>
std::vector<T> regular_rate_sampling(const std::vector<T> &v, size_t rate) {
    if (rate == 0)
        throw std::invalid_argument("rate must be greater than 0");
    if (rate >= v.size())
        throw std::invalid_argument("rate must be less than v.size()");
    std::vector<T> result;
    for (size_t i = 0; i < v.size(); i += rate)
        result.push_back(v[i]);
    return result;
}

template<typename F, class V>
size_t query_time(F f, const V &queries) {
    auto start = timer::now();
    auto cnt = 0;
    for (auto &q: queries)
        cnt += f(q);
    auto stop = timer::now();
    [[maybe_unused]] volatile auto tmp = cnt;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / queries.size();
}