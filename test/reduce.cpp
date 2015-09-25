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

// some internal tests for reduction functions, not part of the official GTest test suite

#include <iostream>
#include <mxx/reduction.hpp>
#include <mxx/shift.hpp>

#include <prettyprint.hpp>

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);

    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    /* code */
    int x = 2*rank+1;
    //int y = mxx::allreduce(x, [](int x, int y){return x+y;}, comm);
    //int y = mxx::allreduce(x, std::max<int>, comm);
    int y = mxx::allreduce(x, std::plus<int>(), comm);
    std::cout << "sum = " << y << std::endl;

    MPI_Datatype ct;
    MPI_Type_contiguous(3, MPI_INT, &ct);
    //std::string str = "hello there!";
    mxx::attr_map<int, std::string>::set(ct, 13, "hi blah blah!");
    mxx::attr_map<int, std::string>::set(ct, 13, "xxxx");
    mxx::attr_map<int, int>::set(ct, 13, 14);
    // TODO: test deletion, thread-safety, etc
    // TODO: what happens if i set some index to different types??
    // TODO: should i normalize to use a fixed type for indexing?

    mxx::attr_map<int, std::function<void()> >::set(ct, 1, [](){std::cout << "Called std::function!" << std::endl;});

    MPI_Datatype dt;
    MPI_Type_dup(ct, &dt);
    mxx::attr_map<int, std::string>::set(dt, 13, "blubb");
    mxx::attr_map<int, int>::set(ct, 13, 15);

    // and get
    std::cout << mxx::attr_map<int, std::string>::get(ct, 13) << ", " << mxx::attr_map<int,int>::get(ct,13) << std::endl;
    std::cout << mxx::attr_map<int, std::string>::get(dt, 13) << ", " << mxx::attr_map<int,int>::get(dt,13) << std::endl;

    // try executing function:
    mxx::attr_map<int, std::function<void()> >::get(ct, 1)();

    MPI_Type_free(&ct);
    //MPI_Type_free(&dt);

    // check mxx builtin
    std::cout << "builtin int: " << mxx::is_builtin_type<int>::value << std::endl;
    std::cout << "builtin size_t: " << mxx::is_builtin_type<size_t>::value << std::endl;
    std::cout << "builtin tuple: " << mxx::is_builtin_type<std::tuple<int,int> >::value << std::endl;

    //int i = rank;
    //std::tuple<int, double> t(rank, 3.14*rank);
    //mxx::future<std::tuple<int, double> > t2 = mxx::async_left_shift(t);
    //int j; double d;
    //std::tie(j, d) = t2.get();
    //std::cout << "rank " << i << " received: " << j << "," << d << std::endl;

    std::vector<int> vec_send = {13, 14, 15};
    // self send
    if (mxx::comm().rank() == 0) {
        mxx::future<void> fut = mxx::async_send(vec_send, 0, 1);
        std::vector<int> vec = mxx::recv<std::vector<int>>(0, 1);
        fut.wait();
        std::cout << "received vec: " << vec << std::endl;
    }

    {
    typedef std::tuple<std::array<std::size_t, 4>, unsigned int, unsigned int, unsigned int> tuple_t;
    mxx::datatype<std::array<std::size_t, 4> > dat;
    mxx::datatype<tuple_t> dat2;
    std::cout << "size of the large tuple: " << sizeof(tuple_t) << std::endl;
    MPI_Datatype mpi_dat = dat.type();
    }


    // finalize MPI
    MPI_Finalize();
    return 0;
}
