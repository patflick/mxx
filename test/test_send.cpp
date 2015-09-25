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

#include <mxx/comm.hpp>
#include <gtest/gtest.h>


TEST(MxxComm, SendRecv1) {
    mxx::comm c;

    // send variable sized vector in cyclic order
    std::vector<int> x(c.rank() + 1);
    // receive from next
    mxx::future<std::vector<int>> fvec = c.irecv_vec<int>(((c.rank()+1) % c.size())+1,(c.rank() + 1) % c.size());
    // send to previous
    c.isend(x, (c.rank() + (c.size()-1)) % c.size());
    ASSERT_EQ((c.rank() + 1) % c.size() + 1u, fvec.get().size());

    // send a string message in cyclic order between all processes in the communicator
    mxx::future<std::string> fstr = c.irecv_str(12,(c.rank() + 1) % c.size());
    c.send("Hello World!", (c.rank()+c.size()-1) % c.size());
    std::string s = fstr.get();
    ASSERT_EQ(std::string("Hello World!"), s);

}
