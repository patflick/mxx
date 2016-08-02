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
#include <mxx/sort.hpp>
#include <mxx/bitonicsort.hpp>
#include <mxx/shift.hpp>
#include <mxx/stream.hpp>

#include <vector>
#include <iostream>

#include <cxx-prettyprint/prettyprint.hpp>

TEST(MxxSort, SampleSort1) {
    mxx::comm c;
    // simple sorting test with same number of elements for each processor
    std::vector<int> vec(230);
    srand(c.rank());
    std::generate(vec.begin(), vec.end(), std::rand);

    // sort
    mxx::sort(vec.begin(), vec.end(), std::less<int>(), c);

    // should be locally sorted
    ASSERT_TRUE(std::is_sorted(vec.begin(), vec.end()));

    // first element on each proc should be larger or equal to last one on
    // previous processor
    int prev = mxx::right_shift(vec.back(), c);
    if (c.rank() > 0) {
        ASSERT_TRUE(prev <= vec.front());
    }

    // assume is sorted
    bool sorted = mxx::is_sorted(vec.begin(), vec.end(), std::less<int>(), c);
    ASSERT_EQ(true, sorted);

}

TEST(MxxSort, BitonicSort1) {
    mxx::comm c;
    // simple sorting test with same number of elements for each processor
    std::vector<int> vec(250);
    srand(133*c.rank());
    std::generate(vec.begin(), vec.end(), [](){ return std::rand() % 100;});

    // sort
    mxx::bitonic_sort(vec.begin(), vec.end(), std::less<int>(), c);

    // should be locally sorted
    ASSERT_TRUE(std::is_sorted(vec.begin(), vec.end()));

    // first element on each proc should be larger or equal to last one on
    // previous processor
    int prev = mxx::right_shift(vec.back(), c);
    if (c.rank() > 0) {
        ASSERT_TRUE(prev <= vec.front());
    }

    // assume is sorted
    bool sorted = mxx::is_sorted(vec.begin(), vec.end(), std::less<int>(), c);
    ASSERT_EQ(true, sorted);
}



TEST(MxxSort, SampleSortInbalanced) {
    mxx::comm c;
    srand(c.rank()*13);
    std::vector<std::pair<int,int> > vec(10 + c.rank()*3);

    std::generate(vec.begin(), vec.end(), [](){return std::make_pair(std::rand(), std::rand());});

    // comparator for sorting
    auto paircmp = [](const std::pair<int,int>& x, const std::pair<int, int>& y){
        return x.first < y.first || (x.first == y.first && x.second < y.second);
    };

    // send all elements to master and sort there as "groud-truth"
    std::vector<std::pair<int, int>> truth = mxx::gatherv(vec, 0, c);
    if (c.rank() == 0)
        std::sort(truth.begin(), truth.end(), paircmp);

    // distributed sort
    mxx::sort(vec.begin(), vec.end(), paircmp, c);

    // send to rank 0
    std::vector<std::pair<int,int>> allsorted = mxx::gatherv(vec, 0, c);

    // should be sorted locally
    ASSERT_TRUE(std::is_sorted(vec.begin(), vec.end(), paircmp));
    // should be sorted globally
    ASSERT_TRUE(mxx::is_sorted(vec.begin(), vec.end(), paircmp, c));

    // on rank 0, vectors should be the same
    if (c.rank() == 0) {
        ASSERT_EQ(truth.size(), allsorted.size());
        for (size_t i = 0; i < truth.size(); ++i) {
            EXPECT_EQ(truth[i].first, allsorted[i].first);
            EXPECT_EQ(truth[i].second, allsorted[i].second);
        }
    }
}

TEST(MxxSort, StableSort) {
    mxx::comm c;
    typedef std::pair<int,int> tuple_t;
    std::vector<tuple_t> vec(100+c.rank());
    size_t prefix = c.rank() == 0 ? 0 : (100*(c.rank()) + ((c.rank()+1)*c.rank())/2);
    std::srand(c.rank()*7);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = tuple_t(prefix+i, std::rand() % 4);
    }

    auto snd_cmp = [](const tuple_t& x, const tuple_t& y) {
        return std::get<1>(x) < std::get<1>(y);
    };
    // stably sort only by second index
    mxx::stable_sort(vec.begin(), vec.end(), snd_cmp, c);

    // assert it is sorted lexicographically by both (second, first)
    auto full_cmp = [](const tuple_t& x, const tuple_t& y) {
        return std::get<1>(x) < std::get<1>(y) || (std::get<1>(x) == std::get<1>(y) && std::get<0>(x) < std::get<0>(y));
    };
    ASSERT_TRUE(mxx::is_sorted(vec.begin(), vec.end(), full_cmp, c));
}

TEST(MxxSort, Unique) {
    mxx::comm c;
    std::vector<int> vec(100);
    std::srand(13*c.rank());
    int i = 0;
    std::generate(vec.begin(), vec.end(), [&i](){return i++ % 10;});
    mxx::sort(vec.begin(), vec.end());
    std::vector<int>::iterator newend = mxx::unique(vec.begin(), vec.end());
    std::vector<int> unique_els(vec.begin(), newend);

    std::vector<int> all = mxx::allgatherv(unique_els);
    ASSERT_EQ(10ul, all.size());
    for (size_t i = 0; i < all.size(); ++i) {
        ASSERT_EQ((int)i, all[i]);
    }
}
