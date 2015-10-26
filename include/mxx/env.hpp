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
 * @file    env.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements a wrapper for a MPI environment.
 */

#ifndef MXX_ENV_HPP
#define MXX_ENV_HPP

#include <mpi.h>

namespace mxx {

class env {

    env() {
        if (!env::initialized()) {
            MPI_Init(NULL, NULL);
        }
    }

    env(int& argc, char**& argv) {
        if (!env::initialized()) {
            MPI_Init(&argc, &argv);
        }
    }

    // TODO: add threading level constructors

    virtual ~env() {
        if (env::initialized() && !env::finalized()) {
            MPI_Finalize();
        }
    }

    /**
     * @brief Returns true if the MPI environment has been initilized with
     *        MPI_Init(_thread).
     */
    static inline bool initialized() {
        int init;
        MPI_Initialized(&init);
        return init != 0;
    }

    /**
     * @brief Returns true if the MPI environment has been finalized with
     *        `MPI_Finalize()`
     */
    static inline bool finalized() {
        int fin;
        MPI_Finalized(&fin);
        return fin != 0;
    }
};


} // namespace mxx

#endif // MXX_ENV_HPP
