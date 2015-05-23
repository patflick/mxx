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
TEST(MxxScatter, ScatterOne)
{
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

