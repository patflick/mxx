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
 * @file    mpi_tests.cpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   GTest Unit Tests for collective operations.
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include <mxx/collective.hpp>
#include <mxx/utils.hpp>
#include <cxx-prettyprint/prettyprint.hpp>


// scatter of size 1
TEST(MxxColl, ScatterOne) {
    mxx::comm c = MPI_COMM_WORLD;

    // print node distrbution
    mxx::print_node_distribution(c);

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

    ASSERT_EQ(13, (int)msg1.size());
    ASSERT_EQ(13, (int)msg2.size());
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
    ASSERT_EQ(c.rank()+1, (int)result.size());
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(13*j-23*c.rank(), result[j]);
    }
    if (c.rank() == 0)
        result = mxx::scatterv(&vec[0], sizes, c.rank()+1, 0, c);
    else {
        result.resize(c.rank() + 1);
        mxx::scatterv_recv(&result.front(), c.rank()+1, 0);
    }
    ASSERT_EQ(c.rank()+1, (int)result.size());
    for (int j = 0; j < (int)result.size(); ++j) {
        ASSERT_EQ(13*j-23*c.rank(), result[j]);
    }

    // test convenience functions
    result = mxx::scatterv(vec, sizes, c.rank()+1, 0, c);
    ASSERT_EQ(c.rank()+1, (int)result.size());
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

    ASSERT_EQ(5*(c.size()-c.rank())-3, (int)result.size());
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
        ASSERT_EQ(c.size(), (int)pairs.size());
        for (int i = 0; i < c.size(); ++i)
        {
            ASSERT_EQ(13*i, pairs[i].first);
            ASSERT_EQ(3.141/i, pairs[i].second);
        }
    } else {
        ASSERT_EQ(0, (int)pairs.size());
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
                ASSERT_EQ((unsigned)j*3*i, all[i*size + j]);
            }
        }
    }

    // test convenience functions
    all = mxx::gather(&els[0], size, c.size()-1, MPI_COMM_WORLD);
    if (c.rank() == c.size()-1) {
        ASSERT_EQ(size*c.size(), all.size());
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (int)size; ++j) {
                ASSERT_EQ((unsigned)j*3*i, all[i*size + j]);
            }
        }
    } else {
        ASSERT_EQ(0, (int)all.size());
    }
    // test convenience functions
    all = mxx::gather(els, c.size()-1, MPI_COMM_WORLD);
    if (c.rank() == c.size()-1) {
        ASSERT_EQ(size*c.size(), all.size());
        for (int i = 0; i < c.size(); ++i) {
            for (int j = 0; j < (int)size; ++j) {
                ASSERT_EQ((unsigned)j*3*i, all[i*size + j]);
            }
        }
    } else {
        ASSERT_EQ(0, (int)all.size());
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
        ASSERT_EQ(0u, all.size());
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
        ASSERT_EQ(0u, all.size());
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
        ASSERT_EQ(0u, all.size());
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
        ASSERT_EQ(0u, all.size());
    }
}

TEST(MxxColl, AllgatherOne) {
    mxx::comm c;
    std::pair<int,double> x = std::make_pair(13*c.rank(), 3.141/c.rank());
    // allgather
    std::vector<std::pair<int, double> > pairs = mxx::allgather(x);

    ASSERT_EQ(c.size(), (int)pairs.size());
    for (int i = 0; i < c.size(); ++i)
    {
        ASSERT_EQ(13*i, pairs[i].first);
        ASSERT_EQ(3.141/i, pairs[i].second);
    }
}

TEST(MxxColl, AllgatherGeneral) {
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
            ASSERT_EQ((unsigned)j*3*i, all[i*size + j]);
        }
    }

    // test convenience functions
    all = mxx::allgather(&els[0], size, MPI_COMM_WORLD);
    ASSERT_EQ(size*c.size(), all.size());
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (int)size; ++j) {
            ASSERT_EQ((unsigned)j*3*i, all[i*size + j]);
        }
    }
    // test convenience functions
    all = mxx::allgather(els);
    ASSERT_EQ(size*c.size(), all.size());
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < (int)size; ++j) {
            ASSERT_EQ((unsigned)j*3*i, all[i*size + j]);
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

TEST(MxxColl, All2allOne) {
    mxx::comm c;
    srand(0); // make test reproducable

    // create single message for each processor
    std::vector<std::pair<int, int> > msgs(c.size());
    for (int i = 0; i < c.size(); ++i) {
        msgs[i].first = c.rank() * i;
        msgs[i].second = rand();
    }

    // first all2all
    std::vector<std::pair<int, int> > result(c.size());
    mxx::all2all(&msgs[0], 1, &result[0], c);
    for (int i = 0; i < c.size(); ++i) {
        ASSERT_EQ(i*c.rank(), result[i].first);
        // change the random number
        result[i].second /= 13;
    }

    // send back
    std::vector<std::pair<int, int> > result2 = mxx::all2all(result, c);
    ASSERT_EQ(c.size(), (int)result2.size());
    // check the result
    for (int i = 0; i < c.size(); ++i) {
        ASSERT_EQ(i*c.rank(), result2[i].first);
        ASSERT_EQ(msgs[i].second / 13, result2[i].second);
    }

    // test the last convenience function
    std::vector<std::pair<int, int> > result3 = mxx::all2all(&result2[0], 1);
    // check result
    ASSERT_EQ(c.size(), (int)result3.size());
    for (int i = 0; i < c.size(); ++i) {
        ASSERT_EQ(result[i].first, result3[i].first);
        ASSERT_EQ(result[i].second, result3[i].second);
    }
}

TEST(MxxColl, All2allGeneral) {
    mxx::comm c;

    int size = 230;
    std::vector<int> msgs(size*c.size());
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < size; ++j) {
            msgs[i*size + j] = i*c.rank()*33 - 2*j;
        }
    }

    std::vector<int> result = mxx::all2all(msgs);
    ASSERT_EQ(size*c.size(), (int)result.size());
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < size; ++j) {
            ASSERT_EQ(i*c.rank()*33 - 2*j, result[i*size + j]);
        }
    }
}

TEST(MxxColl, All2allvGeneral) {
    mxx::comm c;

    srand(0); // make test reproducable
    std::vector<size_t> send_counts(c.size());
    std::vector<size_t> recv_counts(c.size());
    std::vector<std::pair<int, double> > msgs;
    for (int i = 0; i < c.size(); ++i) {
        send_counts[i] = 2*c.rank() + 3*(c.size()-i-1);
        recv_counts[i] = 2*i + 3*(c.size()-c.rank()-1);
        for (size_t j = 0; j < send_counts[i]; ++j) {
            msgs.push_back(std::make_pair(c.rank()*i, 1.0 / rand()));
        }
    }
    size_t recv_size = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

    // send one way
    std::vector<std::pair<int, double> > result = mxx::all2allv(msgs, send_counts, recv_counts, c);

    ASSERT_EQ(recv_size, result.size());
    std::vector<std::pair<int, double> >::iterator it = result.begin();
    for (int i = 0; i < c.size(); ++i) {
        for (size_t j = 0; j < recv_counts[i]; ++j) {
            ASSERT_EQ(i*c.rank(), it->first);
            // square the double:
            it->second *= it->second;
            it++;
        }
    }

    // send back
    std::vector<std::pair<int, double> > result2 = mxx::all2allv(&result[0], recv_counts, send_counts);

    // check that the msgs and result2 are same in first and squared in second
    ASSERT_EQ(msgs.size(), result2.size());
    for (size_t i = 0; i < result2.size(); ++i) {
        ASSERT_EQ(msgs[i].first, result2[i].first);
        ASSERT_EQ(msgs[i].second*msgs[i].second, result2[i].second);
    }
}

TEST(MxxColl, All2allvUnknownSize) {
    mxx::comm c;

    std::vector<size_t> send_counts(c.size());
    std::vector<char> msgs;
    for (int i = 0; i < c.size(); ++i) {
        send_counts[i] = (i + c.rank()) % c.size() + 3;
        for (size_t j = 0; j < send_counts[i]; ++j) {
            msgs.push_back('A'+i+j+c.rank());
        }
    }

    // send without knowing the receive count
    std::vector<char> result = mxx::all2allv(msgs, send_counts, c);

    // get the expected receive counts
    std::vector<size_t> actual_recv_count = mxx::all2all(send_counts);
    size_t recv_size = std::accumulate(actual_recv_count.begin(), actual_recv_count.end(), static_cast<size_t>(0));

    ASSERT_EQ(recv_size, result.size());
    std::vector<char>::iterator it = result.begin();
    for (int i = 0; i < c.size(); ++i) {
        for (size_t j = 0; j < actual_recv_count[i]; ++j) {
            ASSERT_EQ((char)('A'+i+c.rank()+j), *it);
            it++;
        }
    }
}
