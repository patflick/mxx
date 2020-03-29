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

#include <cxx-prettyprint/prettyprint.hpp>

template <typename Container, typename Func>
void test_distribute(size_t size, Func gen, const mxx::comm& c) {
    size_t prefix = mxx::exscan(size, c);
    size_t total_size = mxx::allreduce(size, c);

    typedef typename Container::value_type T;
    Container vec;
    vec.resize(size);
    std::srand(13*c.rank());
    for (size_t i = 0; i < size; ++i) {
        vec[i] = gen(prefix+i);
    }

    // stable distribute
    Container eq_distr = mxx::distribute(vec, c);
    mxx::distribute_inplace(vec, c);

    // get expected size
    size_t eq_size = total_size / c.size();
    if ((size_t)c.rank() < total_size % c.size())
        eq_size+=1;
    size_t eq_prefix = mxx::exscan(eq_size, c);

    ASSERT_EQ(eq_size, eq_distr.size());
    ASSERT_EQ(eq_size, vec.size());

    mxx::sort(vec.begin(), vec.end(), c);
    mxx::sort(eq_distr.begin(), eq_distr.end(), c);

    for (size_t i = 0; i < eq_distr.size(); ++i) {
        T expected = gen(eq_prefix+i);
        ASSERT_EQ(expected, eq_distr[i]);
        ASSERT_EQ(expected, vec[i]);
    }
}

template <typename Container, typename Func>
void test_stable_distribute(size_t size, Func gen, const mxx::comm& c) {
    size_t prefix = mxx::exscan(size, c);
    size_t total_size = mxx::allreduce(size, c);

    typedef typename Container::value_type T;
    Container vec;
    vec.resize(size);
    std::srand(13*c.rank());
    for (size_t i = 0; i < size; ++i) {
        vec[i] = gen(prefix+i);
    }

    // stable distribute
    Container eq_distr = mxx::stable_distribute(vec, c);
    mxx::stable_distribute_inplace(vec, c);

    // get expected size
    size_t eq_size = total_size / c.size();
    if ((size_t)c.rank() < total_size % c.size())
        eq_size+=1;
    size_t eq_prefix = mxx::exscan(eq_size, c);

    ASSERT_EQ(eq_size, eq_distr.size());
    ASSERT_EQ(eq_size, vec.size());

    for (size_t i = 0; i < eq_distr.size(); ++i) {
        T expected = gen(eq_prefix+i);
        ASSERT_EQ(expected, eq_distr[i]);
        ASSERT_EQ(expected, vec[i]);
    }
}



TEST(MxxDistribution, DistributePairVector) {
    mxx::comm c;
    // create unequal distribution
    size_t size = std::max<long>(10, 100 - 10 * c.rank());

    auto gen = [](size_t i) {
        return std::pair<int,int>(i, i);
    };

    typedef std::vector<std::pair<int,int>> container_type;
    test_distribute<container_type>(size, gen, c);
    test_stable_distribute<container_type>(size, gen, c);

    size = 0;
    if (c.rank() == c.size()/2) {
        size = 10*c.size();
    }

    test_distribute<container_type>(size, gen, c);
    test_stable_distribute<container_type>(size, gen, c);
}



TEST(MxxDistribution, DistributeVector) {
    mxx::comm c;
    // create unequal distribution
    size_t size = std::max<long>(10, 100 - 10 * c.rank());

    auto gen = [](size_t i) {
        return static_cast<int>(i);
    };


    test_distribute<std::vector<int>>(size, gen, c);
    test_stable_distribute<std::vector<int>>(size, gen, c);

    size = 0;
    if (c.rank() == c.size()/2) {
        size = 10*c.size();
    }

    test_distribute<std::vector<int>>(size, gen, c);
    test_stable_distribute<std::vector<int>>(size, gen, c);

    // create a distribution of total size smaller than
    // the total number of processes and zero elements on the last process
    size = (c.rank() % 2 == 0) ? 1 : 0;
    if (c.is_last()) {
      size = 0;
    }
    // XXX: test_distribute fails with an assertion error in mxx::sort
    // test_distribute<std::vector<int>>(size, gen, c);
    test_stable_distribute<std::vector<int>>(size, gen, c);
}


TEST(MxxDistribution, DistributeString) {
    mxx::comm c;
    // create unequal distribution
    size_t size = 5 + 7*c.rank();

    auto gen = [](size_t i) {
        char c = static_cast<char>(i % 256);
        if (c == '\0')
            ++c;
        return c;
    };

    test_stable_distribute<std::string>(size, gen, c);

    size = 0;
    if (c.rank() == 0) {
        size = 10*c.size();
    }

    test_stable_distribute<std::string>(size, gen, c);
}

TEST(MxxDistribution, DistributeEmpty) {
}
