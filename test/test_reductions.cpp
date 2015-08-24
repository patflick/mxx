/**
 * @file    test_reductions.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   GTest Unit Tests for mxx reductions
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
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
        mxx::custom_op<float> o(std::min<float>);
        EXPECT_EQ(MPI_FLOAT, o.get_type());
        EXPECT_EQ(MPI_MIN, o.get_op());
    }
    {
        mxx::custom_op<int> o(std::max<int>);
        EXPECT_EQ(MPI_INT, o.get_type());
        EXPECT_EQ(MPI_MAX, o.get_op());
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
    int y = mxx::reduce(x, c.size()/2, std::min<int>, c);
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


TEST(MxxReduce, AllReduceVec) {
    mxx::comm c;

    int n = 10; // numbers per rank

    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i;
    }

    std::vector<int> w = mxx::allreduce(v);
    ASSERT_EQ(n, w.size());
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(c.size()*i, w[i]);
    }

    w = mxx::allreduce(v, [](int x, int y) { return x+y; }, c.split(c.rank() % 2));
    int mysize = c.size() / 2;
    if (c.size() % 2 == 1 && c.rank() % 2 == 0)
        ++mysize;
    ASSERT_EQ(mysize, c.split(c.rank() % 2).size());
    ASSERT_EQ(n, w.size());
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(mysize*i, w[i]);
    }
}


// TODO: test for BIG MPI calls
// TODO: test for vector reduce
// TODO: test for simple all_of/some_of etc reductions
