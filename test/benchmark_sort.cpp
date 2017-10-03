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
#include <mxx/sort.hpp>
#include <mxx/timer.hpp>

typedef int element_t;

class RandInput
{
private:
    int mod;
public:
    RandInput(int seed = 0, int mod = 100) : mod(mod)
    {
        std::srand(seed);
    }

    element_t operator()()
    {
        return 1.13 * (std::rand() % mod);
    }
};

void time_samplesort(std::size_t input_size, MPI_Comm comm)
{
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    mxx::timer t;
    mxx::blk_dist part(input_size, p, rank);

    // generate local input
    std::size_t local_size = part.local_size();
    std::vector<element_t> local_els(local_size);
    RandInput rand_input(rank, 4);
    std::generate(local_els.begin(), local_els.end(), rand_input);
    // sort
    MPI_Barrier(comm);
    double start = t.elapsed();
    mxx::sort(local_els.begin(), local_els.end(), std::less<element_t>(), comm);
    MPI_Barrier(comm);
    double duration = t.elapsed() - start;
    if (rank == 0)
        // print time taken in csv format
        std::cout << p << ";" << input_size << ";" << duration << std::endl;
    // check if input is sorted
    if (!mxx::is_sorted(local_els.begin(), local_els.end(), std::less<element_t>(), comm))
    {
        std::cerr << "ERROR: Output is not sorted!" << std::endl;
        exit(1);
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
    time_samplesort(input_size, comm);

    // finalize MPI
    MPI_Finalize();
    return 0;
}
