
#include <mxx/comm.hpp>
#include <iostream>
#include <cxx-prettyprint/prettyprint.hpp>

struct Blah {
    const int& x;
};

Blah get(const int& x) {
    Blah b{x};
    return b;
}

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);
    mxx::comm c;

    // send variable sized vector in cyclic order
    std::vector<int> x(c.rank() + 1);
    mxx::future<std::vector<int>> fvec = c.irecv_vec<int>(((c.rank()+1) % c.size())+1,(c.rank() + 1) % c.size());
    c.isend(x, (c.rank() + (c.size()-1)) % c.size());
    std::cout << "On rank " << c.rank() << " got: " << fvec.get() << std::endl;

    // send a string message in cyclic order between all processes in the communicator
    mxx::future<std::string> fstr = c.irecv_str(12,(c.rank() + 1) % c.size());
    c.send("Hello World!", (c.rank()+c.size()-1) % c.size());
    std::cout << "On rank " << c.rank() << " received: " << fstr.get() << std::endl;

    // finalize MPI
    MPI_Finalize();
    return 0;
}
