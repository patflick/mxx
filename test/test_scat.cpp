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
#include <cxx-prettyprint/prettyprint.hpp>


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


TEST(MxxColl, ScattervGeneral) {
    mxx::comm c = MPI_COMM_WORLD;

    std::vector<int> vec;
    std::vector<size_t> sizes;

    // scatter from 0
    if (c.rank() == 0) {
        sizes.resize(c.size());
        for (int i = 0; i < c.size(); ++i) {
            sizes[i] = i+1;
            for (size_t j = 0; j < (size_t)(i+1); ++j)
            {
                vec.push_back(2*j + 3*i);
            }
        }
    }

    // test general case
    std::vector<int> result(c.rank()+1);
    mxx::scatterv(&vec[0], sizes, &result.front(), c.rank()+1, 0, c);
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(2*j+3*c.rank(), result[j]);
    }
}

TEST(MxxColl, ScattervConvenience) {
    mxx::comm c = MPI_COMM_WORLD;

    std::vector<int> vec;
    std::vector<size_t> sizes;

    // scatter from 0
    if (c.rank() == 0) {
        sizes.resize(c.size());
        for (int i = 0; i < c.size(); ++i) {
            sizes[i] = i+1;
            for (size_t j = 0; j < (size_t)(i+1); ++j)
            {
                vec.push_back(13*j + -23*i);
            }
        }
    }

    // test general case with recv
    std::vector<int> result(c.rank()+1);
    if (c.rank() == 0)
        mxx::scatterv(&vec[0], sizes, &result.front(), c.rank()+1, 0, c);
    else
        result = mxx::scatterv_recv<int>(c.rank()+1, 0, MPI_COMM_WORLD);
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(13*j+-23*c.rank(), result[j]);
    }

    // test convenience functions
    result = mxx::scatterv(&vec[0], sizes, c.rank()+1, 0, c);
    ASSERT_EQ(c.rank()+1, result.size());
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(13*j-23*c.rank(), result[j]);
    }
    if (c.rank() == 0)
        result = mxx::scatterv(&vec[0], sizes, c.rank()+1, 0, c);
    else {
        result.resize(c.rank() + 1);
        mxx::scatterv_recv(&result.front(), c.rank()+1, 0);
    }
    ASSERT_EQ(c.rank()+1, result.size());
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(13*j-23*c.rank(), result[j]);
    }

    // test convenience functions
    result = mxx::scatterv(vec, sizes, c.rank()+1, 0, c);
    ASSERT_EQ(c.rank()+1, result.size());
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(13*j-23*c.rank(), result[j]);
    }
}

// test scatterv variants for which the recv_size is unknown
// on non-root processes
TEST(MxxColl, ScattervUnknownSize) {
    mxx::comm c;
    std::vector<int> vec;
    std::vector<size_t> sizes;

    // scatter from last process
    if (c.rank() == c.size()-1) {
        sizes.resize(c.size());
        for (int i = 0; i < c.size(); ++i) {
            size_t size = 5*(c.size()-i)-3;
            sizes[i] = size;
            for (size_t j = 0; j < size; ++j) {
                vec.push_back(-2340*(int)j + 444*i);
            }
        }
    }

    std::vector<int> result;
    if (c.rank() == c.size()-1) {
        result = mxx::scatterv(vec, sizes, c.size()-1, c);
    } else {
        result = mxx::scatterv_recv<int>(c.size()-1);
    }

    ASSERT_EQ(5*(c.size()-c.rank())-3, result.size());
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(-2340*j+444*c.rank(), result[j]);
    }
}


TEST(MxxColl, GatherOne) {
    mxx::comm c;
    std::pair<int,double> x = std::make_pair(13*c.rank(), 3.141/c.rank());
    // gather to process 0
    std::vector<std::pair<int, double> > pairs = mxx::gather(x, 0);

    if (c.rank() == 0) {
        ASSERT_EQ(c.size(), pairs.size());
        for (int i = 0; i < c.size(); ++i)
        {
            ASSERT_EQ(13*i, pairs[i].first);
            ASSERT_EQ(3.141/i, pairs[i].second);
        }
    } else {
        ASSERT_EQ(0, pairs.size());
    }
}

TEST(MxxColl, GatherGeneral) {
    mxx::comm c;

    size_t size = 13;
    std::vector<unsigned> els(size);
    for (size_t i = 0; i < size; ++i) {
        els[i] = i*3*c.rank();
    }

    std::vector<unsigned> all;
    if (c.rank() == c.size()-1) {
        all.resize(13*c.size());
    }
    // test general function
    mxx::gather(&els[0], size, &all[0], c.size()-1, c);

    if (c.rank() == c.size()-1) {
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (int)size; ++j) {
                ASSERT_EQ(j*3*i, all[i*size + j]);
            }
        }
    }

    // test convenience functions
    all = mxx::gather(&els[0], size, c.size()-1, MPI_COMM_WORLD);
    if (c.rank() == c.size()-1) {
        ASSERT_EQ(size*c.size(), all.size());
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (int)size; ++j) {
                ASSERT_EQ(j*3*i, all[i*size + j]);
            }
        }
    } else {
        ASSERT_EQ(0, all.size());
    }
    // test convenience functions
    all = mxx::gather(els, c.size()-1, MPI_COMM_WORLD);
    if (c.rank() == c.size()-1) {
        ASSERT_EQ(size*c.size(), all.size());
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (int)size; ++j) {
                ASSERT_EQ(j*3*i, all[i*size + j]);
            }
        }
    } else {
        ASSERT_EQ(0, all.size());
    }
}


TEST(MxxColl, GathervGeneral) {
    mxx::comm c;
    std::vector<size_t> recv_sizes;
    // fill elements
    size_t size = 5*(c.rank()+2);
    std::vector<int> els(size);
    for (size_t i = 0; i < size; ++i) {
        els[i] = c.rank()*13 - 42*i;
    }
    std::vector<int> all;
    size_t total_size = 0;
    if (c.rank() == 0) {
        // prepare recv sizes and output vector
        recv_sizes.resize(c.size());
        for (int i = 0; i < c.size(); ++i) {
            recv_sizes[i] = 5*(i+2);
            total_size += recv_sizes[i];
        }
        all.resize(total_size);
    }

    // general gatherv
    mxx::gatherv(&els[0], size, &all[0], recv_sizes, 0, c);
    if (c.rank() == 0) {
        size_t pos = 0;
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (5*(i+2)); ++j) {
                ASSERT_EQ(i*13 - 42*j, all[pos++]);
            }
        }
    }
    // convenience functions
    all = mxx::gatherv(&els[0], size, recv_sizes, 0, c);
    if (c.rank() == 0) {
        ASSERT_EQ(total_size, all.size());
        size_t pos = 0;
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (5*(i+2)); ++j) {
                ASSERT_EQ(i*13 - 42*j, all[pos++]);
            }
        }
    } else {
        ASSERT_EQ(0, all.size());
    }
    // convenience functions
    all = mxx::gatherv(els, recv_sizes, 0);
    if (c.rank() == 0) {
        ASSERT_EQ(total_size, all.size());
        size_t pos = 0;
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (5*(i+2)); ++j) {
                ASSERT_EQ(i*13 - 42*j, all[pos++]);
            }
        }
    } else {
        ASSERT_EQ(0, all.size());
    }
}

TEST(MxxColl, GathervUnknownSize) {
    mxx::comm c;
    // fill elements
    size_t size = 5*(c.rank()+2);
    std::vector<int> els(size);
    for (size_t i = 0; i < size; ++i) {
        els[i] = c.rank()*13 - 42*i;
    }
    size_t total_size = 0;
    if (c.rank() == 0) {
        // prepare recv sizes and output vector
        for (int i = 0; i < c.size(); ++i) {
            total_size += 5*(i+2);
        }
    }

    // gatherv with unknown receive sizes
    std::vector<int> all;
    all = mxx::gatherv(&els[0], size, 0, c);
    if (c.rank() == 0) {
        ASSERT_EQ(total_size, all.size());
        size_t pos = 0;
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (5*(i+2)); ++j) {
                ASSERT_EQ(i*13 - 42*j, all[pos++]);
            }
        }
    } else {
        ASSERT_EQ(0, all.size());
    }
    // convenience function (taking vector)
    all = mxx::gatherv(els, 0);
    if (c.rank() == 0) {
        ASSERT_EQ(total_size, all.size());
        size_t pos = 0;
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (5*(i+2)); ++j) {
                ASSERT_EQ(i*13 - 42*j, all[pos++]);
            }
        }
    } else {
        ASSERT_EQ(0, all.size());
    }
}

TEST(MxxColl, AllgatherOne) {
    mxx::comm c;
    std::pair<int,double> x = std::make_pair(13*c.rank(), 3.141/c.rank());
    // allgather
    std::vector<std::pair<int, double> > pairs = mxx::allgather(x);

    ASSERT_EQ(c.size(), pairs.size());
    for (int i = 0; i < c.size(); ++i)
    {
        ASSERT_EQ(13*i, pairs[i].first);
        ASSERT_EQ(3.141/i, pairs[i].second);
    }
}

TEST(MxxColl, AlllgatherGeneral) {
    mxx::comm c;

    size_t size = 13;
    std::vector<unsigned> els(size);
    for (size_t i = 0; i < size; ++i) {
        els[i] = i*3*c.rank();
    }

    std::vector<unsigned> all;
    all.resize(13*c.size());
    // test general function
    mxx::allgather(&els[0], size, &all[0], c);

    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (int)size; ++j) {
            ASSERT_EQ(j*3*i, all[i*size + j]);
        }
    }

    // test convenience functions
    all = mxx::allgather(&els[0], size, MPI_COMM_WORLD);
    ASSERT_EQ(size*c.size(), all.size());
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (int)size; ++j) {
            ASSERT_EQ(j*3*i, all[i*size + j]);
        }
    }
    // test convenience functions
    all = mxx::allgather(els);
    ASSERT_EQ(size*c.size(), all.size());
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (int)size; ++j) {
            ASSERT_EQ(j*3*i, all[i*size + j]);
        }
    }
}

TEST(MxxColl, AllgathervGeneral) {
    mxx::comm c;
    std::vector<size_t> recv_sizes;
    // fill elements
    size_t size = 5*(c.rank()+2);
    std::vector<int> els(size);
    for (size_t i = 0; i < size; ++i) {
        els[i] = c.rank()*13 - 42*i;
    }
    // prepare recv sizes and output vector
    std::vector<int> all;
    size_t total_size = 0;
    recv_sizes.resize(c.size());
    for (int i = 0; i < c.size(); ++i) {
        recv_sizes[i] = 5*(i+2);
        total_size += recv_sizes[i];
    }
    all.resize(total_size);

    // general gatherv
    mxx::allgatherv(&els[0], size, &all[0], recv_sizes, c);
    size_t pos = 0;
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (5*(i+2)); ++j) {
            ASSERT_EQ(i*13 - 42*j, all[pos++]);
        }
    }
    // convenience functions
    all = mxx::allgatherv(&els[0], size, recv_sizes, c);
    ASSERT_EQ(total_size, all.size());
    pos = 0;
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (5*(i+2)); ++j) {
            ASSERT_EQ(i*13 - 42*j, all[pos++]);
        }
    }
    // convenience functions
    all = mxx::allgatherv(els, recv_sizes);
    ASSERT_EQ(total_size, all.size());
    pos = 0;
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (5*(i+2)); ++j) {
            ASSERT_EQ(i*13 - 42*j, all[pos++]);
        }
    }
}

TEST(MxxColl, AllgathervUnknownSize) {
    mxx::comm c;
    // fill elements
    size_t size = 5*(c.rank()+2);
    std::vector<int> els(size);
    for (size_t i = 0; i < size; ++i) {
        els[i] = c.rank()*13 - 42*i;
    }
    // calculate expected receive size (only for test verification)
    size_t total_size = 0;
    for (int i = 0; i < c.size(); ++i) {
        total_size += 5*(i+2);
    }

    // gatherv with unknown receive sizes
    std::vector<int> all;
    all = mxx::allgatherv(&els[0], size, c);
    ASSERT_EQ(total_size, all.size());
    size_t pos = 0;
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (5*(i+2)); ++j) {
            ASSERT_EQ(i*13 - 42*j, all[pos++]);
        }
    }
    // convenience function (taking vector)
    all = mxx::allgatherv(els);
    ASSERT_EQ(total_size, all.size());
    pos = 0;
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (5*(i+2)); ++j) {
            ASSERT_EQ(i*13 - 42*j, all[pos++]);
        }
    }
}
