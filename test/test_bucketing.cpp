/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <gtest/gtest.h>
#include <mxx/algos.hpp>

#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <cxx-prettyprint/prettyprint.hpp>

#define START() std::chrono::steady_clock::now()
#define STOP_MS(x) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - (x)).count()

// TODO: implement actual testing
TEST(MxxAlgos, BucketingBenchmark) {
    size_t n = 51340200;
    typedef std::tuple<size_t, size_t, size_t> T;
    std::vector<T> input(n);
    std::generate(input.begin(), input.end(), [](){ return std::tuple<size_t, size_t, size_t>(std::rand(), std::rand(), std::rand());});
    std::vector<T> ref(input);

    // create reference by sorting
    //std::sort(ref.begin(), ref.end());

    //auto func = [](const int& i){ return i % 10;};
    //size_t b = 10;
    auto func = [](const T& i){ return std::get<1>(i) % 1000;};
    size_t b = 1000;

    // bucketing via modulo
    auto x = std::chrono::steady_clock::now();
    std::vector<size_t> counts = mxx::bucketing(input, func, b);
    std::cout << "Branch-free Bucketing: " << STOP_MS(x) << "ms" << std::endl;

    // copy input again
    input = ref;
    auto start_inplace = std::chrono::steady_clock::now();
    std::vector<size_t> counts_inplace = mxx::bucketing_inplace(input, func, b);
    std::cout << "Inplace Bucketing: " << STOP_MS(start_inplace) << "ms" << std::endl;

    /*
    input = ref;
    auto start_tony = std::chrono::steady_clock::now();
    std::vector<size_t> counts_tony = mxx::bucketing_tony(input, func, b);
    std::cout << "Tony Bucketing: " << STOP_MS(start_tony) << "ms" << std::endl;
    */
}
