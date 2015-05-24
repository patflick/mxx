/**
 * @file    mpi_tests.cpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   GTest Unit Tests for the parallel MPI code.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include <mxx/collective.hpp>


// scatter of size 1
TEST(MxxColl, ScatterOne) {
    mxx::comm c = MPI_COMM_WORLD;

    std::vector<int> vec;
    int my;
    // scatter from 0
    if (c.rank() == 0) {
        vec.resize(c.size());
        for (int i = 0; i < c.size(); ++i) {
            vec[i] = 3*i*i;
        }
    }
    my = mxx::scatter_one(vec, 0, c);
    ASSERT_EQ(3*c.rank()*c.rank(), my);

    // scatter from last process using custom std::pair type
    std::pair<int, double> result;
    if (c.rank() == c.size()-1) {
        std::vector<std::pair<int, double> > vec2(c.size());
        for (int i = 0; i < c.size(); ++i) {
            vec2[i] = std::make_pair(-2*i, 3.14*i*i);
        }
        result = mxx::scatter_one(vec2, c.size()-1, c);
    } else {
        // use separate receive function
        result = mxx::scatter_one_recv<std::pair<int, double> >(c.size()-1, c);
    }
    ASSERT_EQ(-2*c.rank(), result.first);
    ASSERT_EQ(3.14*c.rank()*c.rank(), result.second);
}


TEST(MxxColl, ScatterUnknownSize) {
    mxx::comm c = MPI_COMM_WORLD;

    std::vector<int> vec;
    std::vector<int> msg1;
    std::vector<int> msg2;
    // scatter from 0
    if (c.rank() == 0) {
        size_t msgsize = 13;
        vec.resize(c.size()*msgsize);
        for (int i = 0; i < c.size(); ++i) {
            for (size_t j = 0; j < msgsize; ++j)
            {
                vec[i*msgsize + j] = -2*j + i;
            }
        }
        // scatter the same thing twice
        msg1 = mxx::scatter(vec, 0);
        msg2 = mxx::scatter(vec, 0, c);
    } else {
        msg1 = mxx::scatter_recv<int>(0, c);
        msg2 = mxx::scatter_recv<int>(0);
    }

    ASSERT_EQ(13, msg1.size());
    ASSERT_EQ(13, msg2.size());
    for (int j = 0; j < 13; ++j) {
        ASSERT_EQ(-2*j+c.rank(), msg1[j]);
        ASSERT_EQ(-2*j+c.rank(), msg2[j]);
    }
}

TEST(MxxColl, ScatterGeneral) {
    mxx::comm c = MPI_COMM_WORLD;

    std::vector<int> vec;
    size_t msgsize = 5;

    // scatter from 0
    if (c.rank() == 0) {
        vec.resize(c.size()*msgsize);
        for (int i = 0; i < c.size(); ++i) {
            for (size_t j = 0; j < msgsize; ++j)
            {
                vec[i*msgsize + j] = 2*j + 3*i;
            }
        }
    }
    std::vector<int> result(msgsize);
    mxx::scatter(&vec[0], msgsize, &result.front(), 0, c);
    for (int j = 0; j < (int)msgsize; ++j) {
        ASSERT_EQ(2*j+3*c.rank(), result[j]);
    }
}

// TODO: test for BIG MPI calls
