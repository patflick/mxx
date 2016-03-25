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
 * @file    mpi_gtest.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   MPI enabled Google Test framework. Process 0 will collect
 *          the results (and failures) of all tests and output it.
 */
#include <mpi.h>
#include <gtest/gtest.h>
#include "mxx_eventlistener.hpp"
#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <iostream>


int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    mxx::env e(argc, argv);
    mxx::env::set_exception_on_error();
    mxx::comm c = mxx::comm();
    //MPI_Init(&argc, &argv);

    if (c.rank() == 0)
      std::cout << "Running GTEST with MPI using " << c.size() << " processes." << std::endl;

    // set up wrapped test listener
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    ::testing::TestEventListener* default_listener =  listeners.Release(listeners.default_result_printer());
    listeners.Append(new mxx_gtest::MpiTestEventListener(c.rank(), default_listener));

    // running tests
    result = RUN_ALL_TESTS();
    if (c.rank() != 0)
      result = 0;

    // clean up MPI
    //MPI_Finalize();

    // return good status no matter what
    return result;
}
