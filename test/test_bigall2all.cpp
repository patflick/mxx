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
#include <mpi.h>

#include <vector>
#include <iostream>

#include <mxx/partition.hpp>
#include <mxx/collective.hpp>

typedef int element_t;
typedef int count_t;

std::size_t rand_around(std::size_t n, std::size_t fraction = 10)
{
    return n - n/fraction + (rand() % (2*(n/fraction)));
}

void test_all2all(std::size_t input_size, MPI_Comm comm)
{
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    // get local size
    mxx::blk_dist part(input_size, p, rank);
    // generate local input
    std::size_t local_size = part.local_size();

    // create "random" send counts
    std::vector<count_t> send_counts(p);
    mxx::blk_dist local_part(local_size, p, rank);
    for (int i = 0; i < p-1; ++i)
    {
        std::size_t s = rand_around(local_part.local_size(), 10);
        send_counts[i] = std::min(s, local_size);
        local_size -= s;
    }
    send_counts[p-1] = local_size;
    local_size = part.local_size();

    // fill in data
    std::vector<element_t> local_els(local_size);
    std::size_t k = 0;
    for (int i = 0; i < p; ++i)
    {
        for (count_t j = 0; j < send_counts[i]; ++j)
        {
            local_els[k++] = i;
        }
    }

    // execute all2all
    std::vector<count_t> expected_recv = mxx::all2all(send_counts);
    std::vector<element_t> results = mxx::all2allv(local_els, send_counts, comm);

    // check all my elements
    count_t exp_recv_n = std::accumulate(expected_recv.begin(), expected_recv.end(), 0);
    if (results.size() != (std::size_t)exp_recv_n){
        std::cerr << "[ERROR]: Received data volume incorrect: Expected: " << exp_recv_n << ", received: " << results.size() << std::endl;
    }
    for (std::size_t i = 0; i < results.size(); ++i)
    {
        if (results[i] != rank)
        {
            std::cerr << "[ERROR]: Received data wrong!" << std::endl;
            break; // break since the data volume might be very large
        }
    }
}

void print_usage()
{
    std::cerr << "Usage:  ./test_sort -n <input_size>" << std::endl;
    exit(1);
}

void my_mpi_errorhandler(MPI_Comm* comm, int* errorcode, ...)
{
    // throw exception, enables gdb stack trace analysis
    throw std::runtime_error("Shit: mpi fuckup");
}

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);

    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // set custom error handler (for debugging with working stack-trace on gdb)
    MPI_Errhandler errhandler;
    MPI_Errhandler_create(&my_mpi_errorhandler, &errhandler);
    MPI_Errhandler_set(comm, errhandler);


    // parse input
    std::size_t input_size = 0;
    if (argc < 3)
        print_usage();
    if (argv[1][0] != '-')
        print_usage();

    switch(argv[1][1])
    {
        case 'n':
            // set input_size
            input_size = atol(argv[2]);
            break;
        case 'm':
            // set input_size
            input_size = static_cast<std::size_t>(p*atol(argv[2]));
            break;
        default:
            print_usage();
    }

    if (input_size == 0)
        print_usage();

    // run algorithm
    MPI_Barrier(comm);
    test_all2all(input_size, comm);

    // finalize MPI
    MPI_Finalize();
    return 0;
}
