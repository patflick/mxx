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

/**
 * @file    test_reductions.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   GTest Unit Tests for mxx reductions
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include <mxx/comm.hpp>
#include <mxx/reduction.hpp>


// test internal details of custom ops

TEST(MxxImpl, CustomOp) {
    // test C++ functors (postive tests)
    {
        std::plus<int> x;
        mxx::custom_op<int> o(x);
        EXPECT_EQ(MPI_INT, o.get_type());
        EXPECT_EQ(MPI_SUM, o.get_op());
    }
    {
        std::multiplies<double> x;
        mxx::custom_op<double> o(x);
        EXPECT_EQ(MPI_DOUBLE, o.get_type());
        EXPECT_EQ(MPI_PROD, o.get_op());
    }
    {
        std::bit_or<char> x;
        mxx::custom_op<char> o(x);
        EXPECT_EQ(MPI_CHAR, o.get_type());
        EXPECT_EQ(MPI_BOR, o.get_op());
    }
    {
        std::bit_xor<long> x;
        mxx::custom_op<long> o(x);
        EXPECT_EQ(MPI_LONG, o.get_type());
        EXPECT_EQ(MPI_BXOR, o.get_op());
    }
    {
        std::bit_and<unsigned char> x;
        mxx::custom_op<unsigned char> o(x);
        EXPECT_EQ(MPI_UNSIGNED_CHAR, o.get_type());
        EXPECT_EQ(MPI_BAND, o.get_op());
    }
    {
        std::logical_or<unsigned int> x;
        mxx::custom_op<unsigned int> o(x);
        EXPECT_EQ(MPI_UNSIGNED, o.get_type());
        EXPECT_EQ(MPI_LOR, o.get_op());
    }
    {
        std::logical_and<unsigned short> x;
        mxx::custom_op<unsigned short> o(x);
        EXPECT_EQ(MPI_UNSIGNED_SHORT, o.get_type());
        EXPECT_EQ(MPI_LAND, o.get_op());
    }
    // test std::min and std::max
    {
        mxx::custom_op<float> o(static_cast<const float&(*)(const float&, const float&)>(&std::min<float>));
        EXPECT_EQ(MPI_FLOAT, o.get_type());
        EXPECT_EQ(MPI_MIN, o.get_op());
    }
    {
        mxx::custom_op<int> o(static_cast<const int&(*)(const int&, const int&)>(&std::max<int>));
        EXPECT_EQ(MPI_INT, o.get_type());
        EXPECT_EQ(MPI_MAX, o.get_op());
    }
    {
        mxx::max<int> x;
        mxx::custom_op<int> o(x);
        EXPECT_EQ(MPI_INT, o.get_type());
        EXPECT_EQ(MPI_MAX, o.get_op());
    }
    {
        mxx::min<int> x;
        mxx::custom_op<int> o(x);
        EXPECT_EQ(MPI_INT, o.get_type());
        EXPECT_EQ(MPI_MIN, o.get_op());
    }
}

template <typename T>
struct mymax {
    T operator()(const T x, T& y){
        if (x > y)
            return x;
        else
            return y;
    }
};

int mymin(int x, int y) {
    if (x < y)
        return x;
    else
        return y;
}

// scatter of size 1
TEST(MxxReduce, ReduceOne) {
    mxx::comm c = mxx::comm();

    // test min
    int x = -13*(c.size() - c.rank());
    int y = mxx::reduce(x, c.size()/2, mxx::min<int>(), c);
    if (c.rank() == c.size()/2) {
        ASSERT_EQ(-13*c.size(), y);
    } else {
        ASSERT_EQ(0, y);
    }
    // test sum
    int z = mxx::reduce(3, 0, std::plus<int>(), c);
    if (c.rank() == 0) {
        ASSERT_EQ(3*c.size(), z);
    } else {
        ASSERT_EQ(0, z);
    }
    // test lambda op
    int g = mxx::reduce(2, c.size()-1, [](int i, int j){return i+j;});
    if (c.rank() == c.size()-1) {
        ASSERT_EQ(2*c.size(), g);
    } else {
        ASSERT_EQ(0, g);
    }
    // test functor
    float m = mxx::reduce(1.3333f*c.rank(), 0, mymax<float>(), c);
    if (c.rank() == 0) {
        ASSERT_EQ(1.3333f*(c.size()-1), m);
    } else {
        ASSERT_EQ(0, m);
    }
    // test own function pointer
    int v = c.rank() + 1;
    int u = mxx::reduce(v, 0, mymin, c);
    if (c.rank() == 0) {
        ASSERT_EQ(1, u);
    } else {
        ASSERT_EQ(0, u);
    }
}

TEST(MxxReduce, ReduceVec) {
    mxx::comm c;
    int n = 13;
    std::vector<int> v(n);

    for (int i = 0; i < n; ++i) {
        v[i] = c.rank() + i;
    }

    int ranksum = (c.size() * (c.size()-1)) / 2;

    std::vector<int> w = mxx::reduce(v, c.size()/2, c);
    if (c.rank() == c.size()/2) {
        ASSERT_EQ(n, (int)w.size());
        for (int i = 0; i < n; ++i) {
            ASSERT_EQ(ranksum + i*c.size(), w[i]);
        }
    } else {
        ASSERT_EQ(0u, w.size());
    }
}



TEST(MxxReduce, AllReduceVec) {
    mxx::comm c;

    int n = 10; // numbers per rank

    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i;
    }

    std::vector<int> w = mxx::allreduce(v);
    ASSERT_EQ(n, (int)w.size());
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(c.size()*i, w[i]);
    }

    w = mxx::allreduce(v, [](int x, int y) { return x+y; }, c.split(c.rank() % 2));
    int mysize = c.size() / 2;
    if (c.size() % 2 == 1 && c.rank() % 2 == 0)
        ++mysize;
    ASSERT_EQ(mysize, c.split(c.rank() % 2).size());
    ASSERT_EQ(n, (int)w.size());
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(mysize*i, w[i]);
    }
}

TEST(MxxReduce, Scan) {
    mxx::comm c;

    int r = c.rank()*2;
    int g = mxx::scan(r);
    ASSERT_EQ(c.rank()*(c.rank()+1), g);
    int m = mxx::exscan(r, mxx::max<int>());
    if (r != 0) {
        ASSERT_EQ((c.rank()-1)*2, m);
    }
    int m2 = mxx::exscan(r, mxx::max<int>(), c.split(c.rank() % 3));
    if (c.rank() >= 3) {
        ASSERT_EQ((c.rank()-3)*2, m2);
    }
}

TEST(MxxReduce, GlobalReduce) {
    mxx::comm c;

    // test reduce with zero elements for some processes
    size_t n = 0;
    int presize = 0;
    if (c.rank() % 2 == 0) {
        n = (c.rank()/2+1);
        presize = n*(n-1)/2;
    }
    std::vector<long> local(n);
    for (size_t i = 0; i < n; ++i) {
        local[i] = presize + i + 1;
    }
    long totalsum = mxx::global_reduce(local, [](int x, int y){ return x+y; }, c);
    int nonzero_size = (c.size()+1)/2;
    int num_els = nonzero_size*(nonzero_size+1)/2;
    ASSERT_EQ(num_els*(num_els+1)/2, totalsum);
}

TEST(MxxReduce, GlobalScan) {
    mxx::comm c;
    // test scan with zero elements for some processes
    size_t n = 0;
    int presize = 0;
    if (c.rank() % 2 == 0) {
        n = (c.rank()/2+1);
        presize = n*(n-1)/2;
    }
    std::vector<int> local(n);
    for (size_t i = 0; i < n; ++i) {
        local[i] = presize + i + 1;
    }
    // test inplace scan
    std::vector<int> local_cpy(local);
    mxx::global_scan_inplace(local_cpy.begin(), local_cpy.end(), c);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(local[i]*(local[i]+1)/2, local_cpy[i]);
    }
    // test scan
    //std::vector<int> result(local.size());
    std::vector<int> result = mxx::global_scan(local, [](int x, int y) {return x+y;});
    ASSERT_EQ(local.size(), result.size());
    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(local[i]*(local[i]+1)/2, result[i]);
    }
}

TEST(MxxReduce, GlobalScanMin) {
    mxx::comm c;
    // test scan with min and an array of positive
    // elements sorted in ascending order
    size_t n = c.size();
    std::vector<int> local(n);
    for (size_t i = 0; i < n; ++i) {
        local[i] = (c.rank()*n)+i+1;
    }
    // test inplace scan
    std::vector<int> local_cpy(local);
    mxx::global_scan_inplace(local_cpy.begin(), local_cpy.end(), mxx::min<int>(), c);
    for (size_t i = 0; i < n; ++i) {
        // 1 is both the first as well as the minimum element
        ASSERT_EQ(1, local_cpy[i]);
    }
    // test scan
    std::vector<int> result = mxx::global_scan(local, mxx::min<int>(), c);
    ASSERT_EQ(local.size(), result.size());
    for (size_t i = 0; i < n; ++i) {
        // 1 is both the first as well as the minimum element
        ASSERT_EQ(1, result[i]);
    }
}

TEST(MxxReduce, GlobalExScan) {
    mxx::comm c;
    // test reduce with zero elements for some processes
    size_t n = 0;
    int presize = 0;
    if (c.rank() % 2 == 0) {
        n = (c.rank()/2+1);
        presize = n*(n-1)/2;
    }
    std::vector<int> local(n);
    for (size_t i = 0; i < n; ++i) {
        local[i] = presize + i + 1;
    }
    // test inplace exscan
    std::vector<int> local_cpy(local);
    mxx::global_exscan_inplace(local_cpy, std::plus<int>(), c);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(local[i]*(local[i]-1)/2, local_cpy[i]);
    }
    // test exscan
    std::vector<int> result = mxx::global_exscan(local);
    ASSERT_EQ(local.size(), result.size());
    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(local[i]*(local[i]-1)/2, result[i]);
    }
}

TEST(MxxReduce, MinMaxLoc) {
    mxx::comm c;

    std::pair<double, int> maxloc = mxx::max_element(3.1*c.rank(), c);
    ASSERT_EQ(c.size()-1, maxloc.second);
    ASSERT_EQ(3.1*(c.size()-1), maxloc.first);

    std::pair<int, int> minloc = mxx::min_element(c.rank()+13);
    ASSERT_EQ(13, minloc.first);
    ASSERT_EQ(0, minloc.second);
}

// TODO: test for simple all_of/some_of etc reductions
