mxx
===

`mxx` is a `C++`/`C++11` template library for `MPI`. The main goal of this
library is to provide two things:

1. Simplified, efficient, and type-safe C++ bindings to common MPI operations.
2. A collection of scalable, high-performance standard algorithms for parallel
   distributed memory architectures. We hope to supply most C++ STL
   algorithms for use in an MPI environment. Scalability and efficiency are of
   utmost importance in `mxx`.

As such, `mxx` is targeting use in rapid `C++` and `MPI` algorithm
development, prototyping, and deployment.

Consider a simple example, where you might want to sort tuples `(int key,double
x, double y)` by key `key` in parallel using `MPI`. Doing so in pure C/MPI
requires quite a lot of coding (~100 lines), debugging, and frustration. Thanks
to `mxx` and `C++11`, this becomes as easy as:

```c++
    typedef std::tuple<int, double, double> tuple_type;
    std::vector<tuple_type> data(local_size);
    // define a comparator for the tuple
    auto cmp = [](const tuple_type& x, const tuple_type& y) {
                   return std::get<0>(x) < std::get<0>(y); }

    // fill the vector ...

    // call mxx::sort to do all the heavy lifting:
    mxx::sort(data.begin(), data.end(), cmp, MPI_COMM_WORLD);
```

In the background, `mxx` performs many things, including (but not limited to):

- mapping the `std::tuple` to a MPI type by creating the appropriate MPI
  datatype (i.e., `MPI_Type_struct`).
- distributing the data if not yet done so
- calling `std::sort` as a local base case, in case the communicator consists of a
  single processor, `mxx::sort` will fall-back to `std::sort`
- in case the data size exceeds the infamous `MPI` size limit of `MAX_INT`,
  `mxx` will not fail, but continue to work as expected
- redistributing the data so that it has the same distribution as given in the
  input to `mxx::sort`

### Status

Currently `mxx` is still at very early stages. Its growing organically during
our parallel algorithm development, but we hope that it will some day reach a
point of maturity.

### Competition

To our knowledge, there are two noteworthy, similar libraries available.

1. [**boost::mpi**](https://github.com/boostorg/mpi) offers C++ bindings for a
   large number of MPI functions. As such it corresponds to our main goal *1*.
   Major drawbacks of using *boost::mpi* are the unnecessary overhead of
   *boost::serialization* (especially in terms of memory overhead) and that it
   ceased to be a header only library. Most cluster don't have an installation
   of  *boost* or the available version is ancient. Boost installations and
   dependencies have become a huge mess.
2. [**mpp**](https://github.com/motonacciu/mpp) offers low-overhead C++ bindings
   for MPI point-to-point communication primitives. As such, this solutions
   shows better performance than *boost::mpi*, but was never continued beyond
   point-to-point communication.

## Authors

- Patrick Flick

## Code organization

The implementation is split into multiple headers:

- [`algos.hpp`](algos.hpp) implements sequential algorithms, used only
  internally in `mxx`.
- [`collective.hpp`](collective.hpp) C++ wrappers around MPI collective
  operations. At this time supports `all2all`, `gather/allgather`,
  `scatter`, `reduce/allreduce`, `scan`, `exscan`.
- [`datatypes.hpp`](datatypes.hpp) C++ type mapping to `MPI_Datatype`. This
  supports all basic `C` datatypes and `std::array`, `std::pair`, `std::tuple`,
  and further provides methods for users to supply their own datatype in a
  simplified fashion.
- [`distribution.hpp`](distribution.hpp) implements functions for data
  distribution (e.g., equally distributing data among processors)
- [`file.hpp`](file.hpp) implements some functionality for parallel reading and
  writing of files. There is not much in there yet.
- [`samplesort.hpp`](samplesort.hpp) implements the sample sort algorithm for
  parallel distributed sorting. This is an internal file. For usage of
  `mxx::sort`, see `sort.hpp`.
- [`shift.hpp`](shift.hpp) implements simple wrappers for shift communication.
- [`sort.hpp`](sort.hpp) implements the `mxx::sort` function.
- [`utils.hpp`](utils.hpp) implements some utility functions, such as `gdb`
  interaction and printing of node/rank distribution for large jobs.


## Installation

Since this is a header only library, simply copy and paste the `mxx` folder into
your project, and you'll be all set.

### Dependencies

At this point, `mxx` requires a `C++11` compatible version of `gcc`. Some
functions still rely on `gcc` specific function calls. `mxx` currently works
with `MPI-2` and `MPI-3`. However, some collective operations and sorting will
work on data sizes `>= 2 GB` only with `MPI-3`.

### Compiling

Not necessary. This is a header only library. There is nothing to compile.

At some point, we will add some examples, and then say how these examples can be
built.

## Licensing

TBD
