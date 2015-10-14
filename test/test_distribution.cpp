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
#include <mxx/comm.hpp>
#include <mxx/distribution.hpp>
#include <mxx/reduction.hpp>
#include <mxx/sort.hpp>

#include <vector>
#include <algorithm>
#include <iostream>

TEST(MxxDistribution, StableBlockDistr) {
    mxx::comm c;

    // create unequal distribution
    size_t size = 100 + 10 * c.rank();
    std::vector<std::pair<int,int>> vec(size);

    // initialze with unique ids
    size_t prefix = mxx::exscan(size, c);
    size_t total_size = mxx::allreduce(size, c);
    std::srand(13*c.rank());
    for (size_t i = 0; i < size; ++i) {
        vec[i].first = prefix+i;
        vec[i].second = std::rand();
    }

    std::vector<std::pair<int,int>> eq_distr = mxx::stable_distribute(vec, c);

    size_t eq_size = total_size / c.size();
    if ((size_t)c.rank() < total_size % c.size())
        eq_size+=1;
    size_t eq_prefix = mxx::exscan(eq_size, c);
    auto cmp = [](const std::pair<int,int>& x, const std::pair<int,int>& y){
        return x.first < y.first;
    };
    ASSERT_TRUE(mxx::is_sorted(eq_distr.begin(), eq_distr.end(), cmp, c));
    ASSERT_EQ(eq_size, eq_distr.size());
    for (size_t i = 0; i < eq_distr.size(); ++i) {
        ASSERT_EQ(eq_prefix+i,(size_t)eq_distr[i].first);
    }
}

TEST(MxxDistribution, BlockDistr) {
    mxx::comm c;
    // create unequal distribution
    size_t size = std::max<long>(10, 100 - 10 * c.rank());
    std::vector<int> vec(size);

    // initialze with unique ids
    size_t prefix = mxx::exscan(size, c);
    size_t total_size = mxx::allreduce(size, c);
    std::srand(13*c.rank());
    for (size_t i = 0; i < size; ++i) {
        vec[i] = prefix+i;
    }

    // calculate expected distribution size
    size_t eq_size = total_size / c.size();
    if ((size_t)c.rank() < total_size % c.size())
        eq_size+=1;
    size_t eq_prefix = mxx::exscan(eq_size, c);

    // equally distribute
    mxx::distribute_inplace(vec, c);

    ASSERT_TRUE(mxx::all_of(eq_size == vec.size()));

    // sort equally distributed vector
    mxx::sort(vec.begin(), vec.end());

    // check that all values are still there
    for (size_t i = 0; i < vec.size(); ++i) {
        ASSERT_EQ(eq_prefix+i,(size_t)vec[i]);
    }
}

// TODO: add more tests
