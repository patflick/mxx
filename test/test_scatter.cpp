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


#include <mxx/collective.hpp>


int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);

    /* code */
    mxx::comm c;

    std::vector<int> vec;
    int my;
    if (c.rank() == 0) {
        vec.resize(c.size());
        for (int i = 0; i < c.size(); ++i) {
            vec[i] = 2*i;
        }
    }
    my = scatter_one(vec, 0, c);

    std::cout << "on " << c.rank() << ": my=" << my << std::endl;

    // finalize MPI
    MPI_Finalize();
    return 0;
}
