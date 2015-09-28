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
#include <mxx/shift.hpp>

#include <vector>
#include <iostream>

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

