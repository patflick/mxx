

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
