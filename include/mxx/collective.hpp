/**
 * @file    collective.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @group   collective
 * @brief   Collective operations.
 *
 * @detail
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef MXX_COLLECTIVE_HPP
#define MXX_COLLECTIVE_HPP

#include <mpi.h>
#include <assert.h>
#include <vector>
#include <limits>

// mxx includes
#include "datatypes.hpp"
#include "comm_fwd.hpp"
#include "shift.hpp"

/// main namespace for mxx
namespace mxx {

// TODO: unify this with regular messages
#ifdef MXX_MAX_INT
// set to smaller value for testing
constexpr size_t max_int = MXX_MAX_INT;
#else
/// maximum message size for MPI
constexpr size_t max_int = std::numeric_limits<int>::max();
#endif

static_assert(sizeof(size_t) == sizeof(MPI_Aint), "MPI_Aint must be the same size as size_t");

/*********************************************************************
 *                             Scatter                              *
 *********************************************************************/

/// Implementation details
namespace impl {

/**
 * @brief   Returns the displacements vector needed by MPI_Alltoallv.
 *
 * @param counts    The `counts` array needed by MPI_Alltoallv
 *
 * @return The displacements vector needed by MPI_Alltoallv.
 */
template <typename index_t = int>
std::vector<index_t> get_displacements(const std::vector<index_t>& counts)
{
    // copy and do an exclusive prefix sum
    std::vector<index_t> result(counts);
    // set the total sum to zero
    index_t sum = 0;
    index_t tmp;

    // calculate the exclusive prefix sum
    typename std::vector<index_t>::iterator begin = result.begin();
    while (begin != result.end())
    {
        tmp = sum;
        // assert that the sum will still fit into the index type (MPI default:
        // int)
        assert((std::size_t)sum + (std::size_t)*begin < (std::size_t) std::numeric_limits<index_t>::max());
        sum += *begin;
        *begin = tmp;
        ++begin;
    }
    return result;
}




/**
 * @brief 
 *
 * @tparam T
 * @param msgs
 * @param size
 * @param out
 * @param root
 * @param comm
 */
template <typename T>
void scatter_big(const T* msgs, size_t size, T* out, int root, const mxx::comm& comm = mxx::comm())
{
    mxx::datatype_contiguous<T> dt(size);
    MPI_Scatter(const_cast<T*>(msgs), 1, dt.type(), out, 1, dt.type(), root, comm);
}
} // namespace impl

/**
 * @fn void scatter(const T* msgs, size_t size, T* out, int root, const mxx::comm& comm)
 *
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * Given consecutive data on the process `root` in the range `[msgs, msgs+p*size)`,
 * where `p` is the size (number of processes) in the communicator `comm`.
 * This function sends a each consecutive `size` number of elements to one of
 * the processes in the order the processes are defined in the communicator.
 *
 * The first `size` elements go to the process with rank `0`, the next `size`
 * elements to the process with rank `1`, and so on. The received data is
 * saved into the consecutive memory range `[out, out+size)`.
 * For the process with `rank == root`, the appropriate data segment of the
 * `[msgs, msgs+p*size)` range is copied to the range `[out, out+size)`.
 *
 * @note The data in the range `[msgs, msgs+p*size)` is accessed only on
 *       the process with rank `root`.
 *
 * @note The memory pointed to by `out` must be allocated to at least a size
 *       `size`.
 *
 * @see MPI_Scatter
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. Must be of size `p*size`.
 * @param size      The size per message. This is the amount of elements which
 *                  are sent to each process. Thus `p*size` is scattered, and
 *                  `size` elements are received by each process in the
 *                  communicator.
 * @param out       Pointer to the output data. This has to point to valid
 *                  memory, which can hold at least `size` many elements of
 *                  type `T`.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void scatter(const T* msgs, size_t size, T* out, int root, const mxx::comm& comm = mxx::comm())
{
    if (size*comm.size() >= mxx::max_int) {
        // own scatter for large messages
        impl::scatter_big(msgs, size, out, root, comm);
    } else {
        // regular implementation
        mxx::datatype<T> dt;
        int isize = size;
        MPI_Scatter(const_cast<T*>(msgs), isize, dt.type(),
                    out, isize, dt.type(), root, comm);
    }
}

/***************************
 *  Convenience functions  *
 ***************************/

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * This overload returns a `std::vector` instead of saving the result into
 * a provided `out` pointer.
 *
 * @see mxx::scatter()
 *
 * @tparam T        The type of the data (of each element).
 * @param msgs      The data to be scattered. Must be of size `p*size`.
 * @param size      The size (number of elements) of each message.
 * @param root      The rank of the process which contains and scatters the data.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered messages as a `std::vector`. This returns messages on each process in `comm`.
 */
template <typename T>
std::vector<T> scatter(const T* msgs, size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result(size);
    scatter(msgs, size, &result[0], root, comm);
    return std::move(result);
}

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * This overload returns a `std::vector` instead of saving the result into
 * a provided `out` pointer. Additionally, the messages are given by a std::vector
 * instead of a pointer.
 *
 * @see mxx::scatter()
 *
 * @tparam T        The type of the data (of each element).
 * @param msgs      The data to be scattered as a `std::vector`. Must be of size `p*size`.
 * @param size      The size (number of elements) of each message.
 * @param root      The rank of the process which contains and scatters the data.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered messages as a `std::vector`. This returns messages on each process in `comm`.
 */
template <typename T>
std::vector<T> scatter(const std::vector<T>& msgs, size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    assert(comm.rank() != root || msgs.size() == size*comm.size());
    std::vector<T> result = scatter(&msgs[0], size, root, comm);
    return std::move(result);
}

/**
 * @brief   Receives the data from a scatter operator. This is only for non `root` processes.
 *
 * Only valid for processes which have `rank` not equal to `root`. This receives
 * the data scattered from the root process.
 *
 * @see scatter()
 *
 * @tparam T        The type of the data (of each element).
 * @param size      The size (number of elements) of each message.
 * @param root      The rank of the process which contains and scatters the data.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered messages as a `std::vector`. This returns messages on each process in `comm`.
 */
template <typename T>
std::vector<T> scatter_recv(size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result(size);
    scatter((const T*)nullptr, size, &result[0], root, comm);
    return std::move(result);
}

/************************
 *  Scatter size first  *
 ************************/

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * This version of the function first communicates the sizes, i.e., assumes
 * that the processes other than `root` do not know the size to receive.
 *
 * @see scatter()
 *
 * @tparam T        The type of the data (of each element).
 * @param msgs      The data to be scattered as a `std::vector`. Must be of size `p*size`.
 * @param root      The rank of the process which contains and scatters the data.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered messages as a `std::vector`. This returns messages on each process in `comm`.
 */
template <typename T>
std::vector<T> scatter(const std::vector<T>& msgs, int root, const mxx::comm& comm = mxx::comm()) {
    size_t size = msgs.size() / comm.size();
    assert(comm.rank() != 0 || msgs.size() % comm.size() == 0);
    mxx::datatype<size_t> sizedt;
    MPI_Bcast(&size, 1, sizedt.type(), root, comm);
    // now everybody knows the size
    std::vector<T> result = scatter(msgs, size, root, comm);
    return std::move(result);
}

/**
 * @brief   Receives the data from a scatter operator. This is only for non `root` processes.
 *
 * This version of the function first communicates the sizes, i.e., assumes
 * that the processes other than `root` do not know the size to receive.
 *
 * This has to be paired with the function which first scatters the sizes
 * and then the data.
 *
 * @see scatter()
 *
 * @tparam T        The type of the data (of each element).
 * @param root      The rank of the process which contains and scatters the data.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered messages as a `std::vector`. This returns messages on each process in `comm`.
 */
template <typename T>
std::vector<T> scatter_recv(int root, const mxx::comm& comm = mxx::comm()) {
    size_t size;
    mxx::datatype<size_t> sizedt;
    MPI_Bcast(&size, 1, sizedt.type(), root, comm);
    // now everybody knows the size
    std::vector<T> result = scatter_recv<T>(size, root, comm);
    return std::move(result);
}


/***********************
 *  Scatter of size 1  *
 ***********************/

/**
 * @brief   Scatters elements from the process `root` to all processes in the communicator.
 *
 * This sends one element to each process in the communicator `comm`.
 * The input vector `msgs` has to have size `comm.size()`.
 *
 * @see scatter()
 *
 * @tparam T        The type of the data (of each element).
 * @param msgs      The data to be scattered as a `std::vector`. Must be of size `p`.
 * @param root      The rank of the process which contains and scatters the data.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered message. The value of the single element per process.
 */
template <typename T>
T scatter_one(const std::vector<T>& msgs, int root, const mxx::comm& comm = mxx::comm()) {
    assert(comm.rank() != root || msgs.size() == comm.size());
    T result;
    scatter(&msgs[0], 1, &result, root, comm);
    return result;
}

/**
 * @brief   Receives a single scattered element, scattered from process `root`.
 *
 * This function receives one element via a `scatter()` operation and returns this
 * element. This function has to be paired with the `scatter_one()` function
 * on the `root` process.
 *
 * @see scatter()
 *
 * @tparam T        The type of the data (of each element).
 * @param root      The rank of the process which contains and scatters the data.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered message. The value of the single element per process.
 */
template <typename T>
T scatter_one_recv(int root, const mxx::comm& comm = mxx::comm()) {
    T result;
    scatter((const T*)nullptr, 1, &result, root, comm);
    return result;
}

/*********************************************************************
 *                             Scatter-V                             *
 *********************************************************************/

namespace impl {

/**
 * @brief Implementation of `scatterv()` for messages with elements more than MAX_INT.
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. Must be of size \f$ \sum_{i=0}^{p-1} \texttt{sizes[i]}\f$ (number of elements of type `T`).
 * @param sizes     The size (number of elements) per message per target process.
 *                  This must be a `std::vector` of size `comm.size()`.
 * @param out       Pointer to the output data. This has to point to valid
 *                  memory, which can hold at least `size` many elements of
 *                  type `T`.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void scatterv_big(const T* msgs, const std::vector<size_t>& sizes, T* out, size_t recv_size, int root, const mxx::comm& comm = mxx::comm()) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::requests reqs;
    int tag = 1234; // TODO: handle tags somewhere (as attributes in the comm?)
    if (comm.rank() == root) {
        std::size_t offset = 0;
        for (int i = 0; i < comm.size(); ++i) {
            mxx::datatype_contiguous<T> dt(sizes[i]);
            if (i == root) {
                // copy input into output
                std::copy(msgs+offset, msgs+offset+sizes[i], out);
            } else {
                MPI_Isend(const_cast<T*>(msgs)+offset, 1, dt.type(), i, tag, comm, &reqs.add());
            }
            offset += sizes[i];
        }
    } else {
        // create custom datatype to encapsulate the whole message
        mxx::datatype_contiguous<T> dt(recv_size);
        MPI_Irecv(const_cast<T*>(out), 1, dt.type(), root, tag, comm, &reqs.add());
    }
    reqs.wait();
}
} // namespace impl

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * The `root` process contains `p` messages, one for each other processor.
 * The message to processor `i` has size `sizes[i]`.
 * The messages are given in the consecutive memory range starting at `msgs` as:
 * \f$ \left[\texttt{msgs},\,\texttt{msgs} + \sum_{j=0}^{p-1} \texttt{sizes[j]} \right) \f$
 * where `p` is the number of processors in the communicator `comm`.
 *
 * Thus, message `i` is located in the memory range given by:
 * \f$ \left[\texttt{msgs} + \sum_{j=0}^{i-1} \texttt{sizes[j]},\,\texttt{msgs} + \sum_{j=0}^{i} \texttt{sizes[j]} \right) \f$
 *
 * The first `sizes[0]` elements go to the process with rank `0`, the next
 * `sizes[1]` elements to the process with rank `1`, and so on. The received
 * data is saved into the consecutive memory range `[out, out+recv_size)`.
 *
 * @note The data in `msgs` is accessed only on the process with rank `root` and can be set to `NULL` for all other processes.
 *
 * @note The memory pointed to by `out` must be allocated to at least a size `recv_size`.
 *
 * @see MPI_Scatterv
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. Must be of size \f$ \sum_{i=0}^{p-1} \texttt{sizes[i]}\f$ (number of elements of type `T`).
 * @param sizes     The size (number of elements) per message per target process.
 *                  This must be a `std::vector` of size `comm.size()`.
 * @param out       Pointer to the output data. This has to point to valid
 *                  memory, which can hold at least `size` many elements of
 *                  type `T`.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void scatterv(const T* msgs, const std::vector<size_t>& sizes, T* out, size_t recv_size, int root, const mxx::comm& comm = mxx::comm()) {
    assert(root != comm.rank() || sizes.size() == comm.size());
    // get total send size
    size_t send_size = std::accumulate(sizes.begin(), sizes.end(), 0);
    mxx::datatype<size_t> sizedt;
    MPI_Bcast(&send_size, 1, sizedt.type(), root, comm);
    // check if we need to use the custom BIG scatterv
    if (send_size >= mxx::max_int) {
        // own scatter for large messages
        impl::scatterv_big(msgs, sizes, out, recv_size, root, comm);
    } else {
        // regular implementation using integer counts
        mxx::datatype<T> dt;
        int irecv_size = recv_size;
        if (comm.rank() == root) {
            std::vector<int> send_counts(comm.size());
            std::copy(sizes.begin(), sizes.end(), send_counts.begin());
            std::vector<int> displs = impl::get_displacements(send_counts);
            MPI_Scatterv(const_cast<T*>(msgs), &send_counts[0], &displs[0], dt.type(),
                         out, irecv_size, dt.type(), root, comm);
        } else {
            MPI_Scatterv(NULL, NULL, NULL, MPI_DATATYPE_NULL,
                         out, irecv_size, dt.type(), root, comm);
        }
    }
}

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * Instead of writing the output to a pointer `out`, this overload of `scatterv()`
 * returns the received messages as a `std::vector` of type `T`.
 *
 * @see scatterv()
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. Must be of size \f$ \sum_{i=0}^{p-1} \texttt{sizes[i]}\f$ (number of elements of type `T`).
 * @param sizes     The size (number of elements) per message per target process.
 *                  This must be a `std::vector` of size `comm.size()`.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns         The received message as `std::vector`, with `recv_size` number of elments.
 */
template <typename T>
std::vector<T> scatterv(const T* msgs, const std::vector<size_t>& sizes, size_t recv_size, int root, const mxx::comm& comm = mxx::comm()) {
    assert(root != comm.rank() || sizes.size() == comm.size());
    std::vector<T> result(recv_size);
    scatterv(msgs, sizes, &result[0], recv_size, root, comm);
    return std::move(result);
}

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * Instead of writing the output to a pointer `out`, this overload of `scatterv()`
 * returns the received messages as a `std::vector` of type `T`.
 *
 * This overload takes a `std::vector` as input on the `root` process instead
 * of a pointer.
 *
 * @see scatterv()
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. This `std::vector` must be of size \f$ \sum_{i=0}^{p-1} \texttt{sizes[i]}\f$ (number of elements of type `T`).
 * @param sizes     The size (number of elements) per message per target process.
 *                  This must be a `std::vector` of size `comm.size()`.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns         The received message as `std::vector`, with `recv_size` number of elments.
 */
template <typename T>
std::vector<T> scatterv(const std::vector<T>& msgs, const std::vector<size_t>& sizes, size_t recv_size, int root, const mxx::comm& comm = mxx::comm()) {
    assert(root != comm.rank() || sizes.size() == comm.size());
    std::vector<T> result(recv_size);
    scatterv(&msgs[0], sizes, &result[0], recv_size, root, comm);
    return std::move(result);
}

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * This function overload first scatters the sizes to be expected on the
 * non-root processes (in case the `recv_size` is unknown).
 *
 * Instead of writing the output to a pointer `out`, this overload of `scatterv()`
 * returns the received messages as a `std::vector` of type `T`.
 *
 * @see scatterv()
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. Must be of size \f$ \sum_{i=0}^{p-1} \texttt{sizes[i]}\f$ (number of elements of type `T`).
 * @param sizes     The size (number of elements) per message per target process.
 *                  This must be a `std::vector` of size `comm.size()`.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns         The received message as `std::vector`, with `recv_size` number of elments.
 */
template <typename T>
std::vector<T> scatterv(const T* msgs, const std::vector<size_t>& sizes, int root, const mxx::comm& comm = mxx::comm()) {
    assert(root != comm.rank() || sizes.size() == comm.size());
    size_t recv_size = scatter_one<size_t>(sizes, root, comm);
    std::vector<T> result = scatterv(msgs, sizes, recv_size, root, comm);
    return std::move(result);
}

/**
 * @brief   Scatters data from the process `root` to all processes in the communicator.
 *
 * This function overload first scatters the sizes to be expected on the
 * non-root processes (in case the `recv_size` is unknown).
 *
 * Instead of writing the output to a pointer `out`, this overload of `scatterv()`
 * returns the received messages as a `std::vector` of type `T`.
 *
 * This overload takes a `std::vector` as input on the `root` process instead
 * of a pointer.
 *
 * @see scatterv()
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. This `std::vector` must be of size \f$ \sum_{i=0}^{p-1} \texttt{sizes[i]}\f$ (number of elements of type `T`).
 * @param sizes     The size (number of elements) per message per target process.
 *                  This must be a `std::vector` of size `comm.size()`.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns         The received message as `std::vector`, with `recv_size` number of elments.
 */
template <typename T>
std::vector<T> scatterv(const std::vector<T>& msgs, const std::vector<size_t>& sizes, int root, const mxx::comm& comm = mxx::comm()) {
    assert(root != comm.rank() || sizes.size() == comm.size());
    size_t recv_size = scatter_one<size_t>(sizes, root, comm);
    std::vector<T> result = scatterv(&msgs[0], sizes, recv_size, root, comm);
    return std::move(result);
}

/**
 * @brief   Receives the data from a scatterv operator. This is only for non `root` processes.
 *
 * Only valid for processes which have `rank` not equal to `root`. This receives
 * the data scattered from the root process.
 *
 * @see scatterv()
 *
 * @tparam T        The type of the data.
 * @param out       Pointer to the output data. This has to point to valid
 *                  memory, which can hold at least `size` many elements of
 *                  type `T`.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void scatterv_recv(T* out, size_t recv_size, int root, const mxx::comm& comm = mxx::comm()) {
    scatterv((const T*) nullptr, std::vector<size_t>(), out, recv_size, root, comm);
}

/**
 * @brief   Receives the data from a scatterv operator. This is only for non `root` processes.
 *
 * Only valid for processes which have `rank` not equal to `root`. This receives
 * the data scattered from the root process.
 *
 * @see scatterv()
 *
 * @tparam T        The type of the data.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered messages as a `std::vector`. This returns messages on each process in `comm`.
 */
template <typename T>
std::vector<T> scatterv_recv(size_t recv_size, int root, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result = scatterv((const T*)nullptr, std::vector<size_t>(), recv_size, root, comm);
    return std::move(result);
}

/**
 * @brief   Receives the data from a scatterv operator. This is only for non `root` processes.
 *
 * Only valid for processes which have `rank` not equal to `root`. This receives
 * the data scattered from the root process.
 *
 * This overload first receives the number of elements in a separate `scatter()`
 * in case `recv_size` is not yet known.
 *
 * This has to be paired with the function which first scatters the sizes
 * and then the data.
 *
 * @see scatterv()
 *
 * @tparam T        The type of the data.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @return  The scattered messages as a `std::vector`. This returns messages on each process in `comm`.
 */
template <typename T>
std::vector<T> scatterv_recv(int root, const mxx::comm& comm = mxx::comm()) {
    size_t recv_size = scatter_one_recv<size_t>(root, comm);
    std::vector<T> result = scatterv_recv<T>(recv_size, root, comm);
    return std::move(result);
}


/*********************************************************************
 *                              Gather                               *
 *********************************************************************/

namespace impl {
/**
 * @brief   Implementation of `gather()` for large message sizes (> MAX_INT).
 *
 * @see mxx::gather
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 *                  This must be the same value on all processes.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least `p*size`
 *                  many elements of type `T`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void gather_big(const T* data, size_t size, T* out, int root, const mxx::comm& comm) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::datatype_contiguous<T> dt(size);
    MPI_Gather(const_cast<T*>(data), 1, dt.type(), out, 1, dt.type(), root, comm);
}
} // namespace impl

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * Gathers a same sized data block from every process to the process
 * with rank `root`. The gathered data on `root` will be received
 * into the memory pointed to by `out`, contiguously arranged. The `size`
 * elements from rank 0 will be first, then the `size` elements from rank `1`,
 * etc.
 *
 * The value of `size` must be the same on all processes.
 *
 * @note The memory pointed to by `out` must be allocated to at least a size
 *       `p*size` on the process with `rank == root`. All other processes may
 *       pass invalid or `NULL` pointers.
 *
 * @see MPI_Gather
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 *                  This must be the same value on all processes.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least `p*size`
 *                  many elements of type `T`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void gather(const T* data, size_t size, T* out, int root, const mxx::comm& comm = mxx::comm()) {
    if (size*comm.size() >= mxx::max_int) {
        // use custom implementation for big data
        impl::gather_big(data, size, out, root, comm);
    } else {
        mxx::datatype<T> dt;
        MPI_Gather(const_cast<T*>(data), size, dt.type(), out, size, dt.type(), root, comm);
    }
}

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * This is a convenience function which returns the received data as a
 * `std::vector`, instead of writing into a given memory segment.
 *
 * @see mxx::gather()
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     A `std::vector`. On the `root` process this will contain all
 *              the gathered data. On all other processes, this vector will
 *              be empty.
 */
template <typename T>
std::vector<T> gather(const T* data, size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result;
    if (comm.rank() == root)
        result.resize(size*comm.size());
    gather(data, size, &result[0], root, comm);
    return std::move(result);
}

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * This is a convenience function which returns the received data as a
 * `std::vector`, instead of writing into a given memory segment.
 * This overload takes a `std::vector` as input intead of 
 *
 * @see mxx::gather()
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must have the same size on all processes.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     A `std::vector`. On the `root` process this will contain all
 *              the gathered data. On all other processes, this vector will
 *              be empty.
 */
template <typename T>
std::vector<T> gather(const std::vector<T>& data, int root, const mxx::comm& comm = mxx::comm()) {
    return gather(&data[0], data.size(), root, comm);
}

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * This is a convenience function which gathers a single element of type `T`
 * from each process and returns it as a `std::vector` on the root process.
 *
 * @see mxx::gather()
 *
 * @tparam T        The type of the data.
 * @param x         The value to be gathered.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     A `std::vector`. On the `root` process this will contain all
 *              the gathered data. On all other processes, this vector will
 *              be empty.
 */
template <typename T>
std::vector<T> gather(const T& x, int root, const mxx::comm& comm = mxx::comm()) {
    return gather(&x, 1, root, comm);
}


/*********************************************************************
 *                             Gather-V                              *
 *********************************************************************/

namespace impl {

/**
 * @brief   Implementation for message sizes larger than MAX_INT elements.
 *
 * @see mxx::gatherv()
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least
 *                  \f$ \sum_{i=0}^{p-1} \texttt{recv\_sizes[i]} \f$
 *                  many elements of type `T`.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void gatherv_big(const T* data, size_t size, T* out, const std::vector<size_t>& recv_sizes, int root, const mxx::comm& comm = mxx::comm()) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::requests reqs;
    int tag = 1234; // TODO: handle tags somewhere (as attributes in the comm?)
    if (comm.rank() == root) {
        size_t offset = 0;
        for (int i = 0; i < comm.size(); ++i) {
            mxx::datatype_contiguous<T> dt(recv_sizes[i]);
            if (i == root) {
                // copy input into output
                std::copy(data, data+size, out+offset);
            } else {
                MPI_Irecv(const_cast<T*>(out)+offset, 1, dt.type(), i, tag, comm, &reqs.add());
            }
            offset += recv_sizes[i];
        }
    } else {
        // create custom datatype to encapsulate the whole message
        mxx::datatype_contiguous<T> dt(size);
        MPI_Isend(const_cast<T*>(data), 1, dt.type(), root, tag, comm, &reqs.add());
    }
    reqs.wait();
}
} // namespace impl


/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * Gathers data from all processes onto the `root` process. In contrast to
 * the `gather()` function, this function allows gathering a different
 * number of elements from each process.
 *
 * The `recv_sizes` parameter gives the number of elements gathered from
 * each process in the order as the processes are ordered in the communicator.
 *
 * All gathered data is saved contiguously into the memory pointed to by `out`.
 *
 * The `gatherv()` function as such is the reverse operation for the `scatterv()`
 * function.
 *
 * @note The memory pointed to by `out` must be allocated to at least a size
 *       \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$ on the process
 *       with `rank == root`. All other processes may pass invalid or
 *       `NULL` pointers.
 *
 * @see MPI_Gatherv
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least
 *                  \f$ \sum_{i=0}^{p-1} \texttt{recv\_sizes[i]} \f$
 *                  many elements of type `T`.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void gatherv(const T* data, size_t size, T* out, const std::vector<size_t>& recv_sizes, int root, const mxx::comm& comm = mxx::comm()) {
    size_t total_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    mxx::datatype<size_t> mpi_sizet;
    // tell everybody about the total size
    MPI_Bcast(&total_size, 1, mpi_sizet.type(), root, comm);
    if (total_size >= mxx::max_int) {
        // use custom implementation for large sizes
        impl::gatherv_big(data, size, out, recv_sizes, root, comm);
    } else {
        // use standard MPI_Gatherv
        mxx::datatype<T> dt;
        if (comm.rank() == root) {
            std::vector<int> counts(comm.size());
            std::copy(recv_sizes.begin(), recv_sizes.end(), counts.begin());
            std::vector<int> displs = impl::get_displacements(counts);
            MPI_Gatherv(const_cast<T*>(data), size, dt.type(),
                        out, &counts[0], &displs[0], dt.type(), root, comm);
        } else {
            MPI_Gatherv(const_cast<T*>(data), size, dt.type(),
                        NULL, NULL, NULL, dt.type(), root, comm);
        }
    }
}

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * Gathers data from all processes onto the `root` process. In contrast to
 * the `gather()` function, this function allows gathering a different
 * number of elements from each process.
 *
 * @see mxx::gatherv()
 *
 * This overload returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$ instead of writing the
 * gathered elements into a pointer `out`. On non-root processes, the returned
 * vector will be empty.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector`. On the root process,
 *              this is a vector of size
 *              \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$. On non-root
 *              processes the vector will be empty.
 */
template <typename T>
std::vector<T> gatherv(const T* data, size_t size, const std::vector<size_t>& recv_sizes, int root, const mxx::comm& comm = mxx::comm()) {
    size_t total_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    std::vector<T> result(total_size);
    gatherv(data, size, &result[0], recv_sizes, root, comm);
    return result;
}

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * Gathers data from all processes onto the `root` process. In contrast to
 * the `gather()` function, this function allows gathering a different
 * number of elements from each process.
 *
 * @see mxx::gatherv()
 *
 * This overload returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$ instead of writing the
 * gathered elements into a pointer `out`. On non-root processes, the returned
 * vector will be empty.
 * The input to this overload is also a `std::vector`.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector`. On the root process,
 *              this is a vector of size
 *              \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$. On non-root
 *              processes the vector will be empty.
 */
template <typename T>
std::vector<T> gatherv(const std::vector<T>& data, const std::vector<size_t>& recv_sizes, int root, const mxx::comm& comm = mxx::comm()) {
    return gatherv(&data[0], data.size(), recv_sizes, root, comm);
}

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * This overload assumes that the receive sizes are not known prior to
 * the gather. As a first step, this function gathers the number of elements
 * on all processors to the root prior to then gathering the actual elements.
 *
 * @see mxx::gatherv()
 *
 * This function returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{size@rank(i)} \f$. On non-root processes, the returned
 * vector will be empty.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector` on the root process.
 *              On non-rooti processes the vector will be empty.
 */
template <typename T>
std::vector<T> gatherv(const T* data, size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    // recv_sizes is unknown -> first collect via gather
    std::vector<size_t> recv_sizes = gather(size, root, comm);
    return gatherv(data, size, recv_sizes, root, comm);
}

/**
 * @brief   Gathers data from all processes onto the root process.
 *
 * This overload assumes that the receive sizes are not known prior to
 * the gather. As a first step, this function gathers the number of elements
 * on all processors to the root prior to then gathering the actual elements.
 *
 * @see mxx::gatherv()
 *
 * This function returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{size@rank(i)} \f$. On non-root processes, the returned
 * vector will be empty.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector` on the root process.
 *              On non-rooti processes the vector will be empty.
 */
template <typename T>
std::vector<T> gatherv(const std::vector<T>& data, int root, const mxx::comm& comm = mxx::comm()) {
    return gatherv(&data[0], data.size(), root, comm);
}


/*********************************************************************
 *                           AllGather                               *
 *********************************************************************/

namespace impl {
/**
 * @brief   Implementation of `allgather()` for large message sizes (> MAX_INT).
 *
 * @see mxx::allgather
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 *                  This must be the same value on all processes.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least `p*size`
 *                  many elements of type `T`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void allgather_big(const T* data, size_t size, T* out, const mxx::comm& comm) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::datatype_contiguous<T> dt(size);
    MPI_Allgather(const_cast<T*>(data), 1, dt.type(), out, 1, dt.type(), comm);
}
} // namespace impl

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * Gathers data which is spread across processes onto each process. At the end
 * of the operation, each process has all the data. The number of elements
 * gathered from each process must be the same. In case it is not, use
 * `allgatherv()` instead.
 *
 * The value of `size` must be the same on all processes.
 *
 * @note The memory pointed to by `out` must be allocated to at least a size
 *       `p*size` on all processes.
 *
 * @see MPI_Allgather
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 *                  This must be the same value on all processes.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least `p*size`
 *                  many elements of type `T`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void allgather(const T* data, size_t size, T* out, const mxx::comm& comm = mxx::comm()) {
    if (size*comm.size() >= mxx::max_int) {
        // use custom implementation for big data
        impl::allgather_big(data, size, out, comm);
    } else {
        mxx::datatype<T> dt;
        MPI_Allgather(const_cast<T*>(data), size, dt.type(), out, size, dt.type(), comm);
    }
}

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * This is a convenience function which returns the received data as a
 * `std::vector`, instead of writing into a given memory segment.
 *
 * @see mxx::allgather()
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 *                  This must be the same value on all processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     A `std::vector` containng all the gathered data.
 */
template <typename T>
std::vector<T> allgather(const T* data, size_t size, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result;
    result.resize(size*comm.size());
    allgather(data, size, &result[0], comm);
    return std::move(result);
}

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * This is a convenience function which returns the received data as a
 * `std::vector`, instead of writing into a given memory segment.
 *
 * @see mxx::allgather()
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must have the same size on all processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     A `std::vector` containng all the gathered data.
 */
template <typename T>
std::vector<T> allgather(const std::vector<T>& data, const mxx::comm& comm = mxx::comm()) {
    return allgather(&data[0], data.size(), comm);
}

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * This is a convenience function which gathers a single element of type `T`
 * from each process and returns it as a `std::vector`.
 *
 * @see mxx::allgather()
 *
 * @tparam T        The type of the data.
 * @param x         The value to be gathered.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     A `std::vector` containng all the gathered data.
 */
template <typename T>
std::vector<T> allgather(const T& x, const mxx::comm& comm = mxx::comm()) {
    return allgather(&x, 1, comm);
}


/*********************************************************************
 *                             Gather-V                              *
 *********************************************************************/

namespace impl {

/**
 * @brief   Implementation for message sizes larger than MAX_INT elements.
 *
 * @see mxx::allgatherv()
 *
 */
template <typename T>
void allgatherv_big(const T* data, size_t size, T* out, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::requests reqs;
    int tag = 1234; // TODO: handle tags somewhere (as attributes in the comm?)
    size_t offset = 0;
    for (int i = 0; i < comm.size(); ++i) {
        // send to this rank
        mxx::datatype_contiguous<T> senddt(size);
        MPI_Isend(const_cast<T*>(data), 1, senddt.type(), i, tag, comm, &reqs.add());
        // receive from this rank
        mxx::datatype_contiguous<T> dt(recv_sizes[i]);
        MPI_Irecv(const_cast<T*>(out)+offset, 1, dt.type(), i, tag, comm, &reqs.add());
        offset += recv_sizes[i];
    }
    reqs.wait();
}
} // namespace impl


/**
 * @brief   Gathers data from all processes onto each process.
 *
 * Gathers data from all processes onto each process. At the end
 * of the operation, each process has all the data. In contrast to
 * the `allgather()` function, this function allows gathering a different
 * number of elements from each process.
 *
 * The `recv_sizes` parameter gives the number of elements gathered from
 * each process in the order as the processes are ordered in the communicator.
 *
 * All gathered data is saved contiguously into the memory pointed to by `out`.
 *
 * @note The memory pointed to by `out` must be allocated to at least a size
 *       \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$ on all processes.
 *
 * @see MPI_Allgatherv
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least
 *                  \f$ \sum_{i=0}^{p-1} \texttt{recv\_sizes[i]} \f$
 *                  many elements of type `T`.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void allgatherv(const T* data, size_t size, T* out, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    size_t total_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    assert(recv_sizes.size() == comm.size());
    mxx::datatype<size_t> mpi_sizet;
    if (total_size >= mxx::max_int) {
        // use custom implementation for large sizes
        impl::allgatherv_big(data, size, out, recv_sizes, comm);
    } else {
        // use standard MPI_Gatherv
        mxx::datatype<T> dt;
        std::vector<int> counts(comm.size());
        std::copy(recv_sizes.begin(), recv_sizes.end(), counts.begin());
        std::vector<int> displs = impl::get_displacements(counts);
        MPI_Allgatherv(const_cast<T*>(data), size, dt.type(),
                    out, &counts[0], &displs[0], dt.type(), comm);
    }
}

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * Gathers data from all processes onto each process. At the end
 * of the operation, each process has all the data. In contrast to
 * the `allgather()` function, this function allows gathering a different
 * number of elements from each process.
 *
 * @see mxx::allgatherv()
 *
 * This overload returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$ instead of writing the
 * gathered elements into a pointer `out`. On non-root processes, the returned
 * vector will be empty.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector` of size
 *              \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$.
 */
template <typename T>
std::vector<T> allgatherv(const T* data, size_t size, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    size_t total_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    std::vector<T> result(total_size);
    allgatherv(data, size, &result[0], recv_sizes, comm);
    return result;
}

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * Gathers data from all processes onto each process. At the end
 * of the operation, each process has all the data. In contrast to
 * the `allgather()` function, this function allows gathering a different
 * number of elements from each process.
 *
 * @see mxx::allgatherv()
 *
 * This overload returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$ instead of writing the
 * gathered elements into a pointer `out`. On non-root processes, the returned
 * vector will be empty.
 * The input to this overload is also a `std::vector`.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector` of size
 *              \f$ \sum_{i=0}^{p-1} \texttt{recv_sizes[i]} \f$.
 */
template <typename T>
std::vector<T> allgatherv(const std::vector<T>& data, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    return allgatherv(&data[0], data.size(), recv_sizes, comm);
}

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * This overload assumes that the receive sizes are not known prior to
 * the gather. As a first step, this function gathers the number of elements
 * on all processors to the root prior to then gathering the actual elements.
 *
 * @see mxx::allgatherv()
 *
 * This function returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{size@rank(i)} \f$.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector`.
 */
template <typename T>
std::vector<T> allgatherv(const T* data, size_t size, const mxx::comm& comm = mxx::comm()) {
    // recv_sizes is unknown -> first collect via gather
    std::vector<size_t> recv_sizes = allgather(size, comm);
    return allgatherv(data, size, recv_sizes, comm);
}

/**
 * @brief   Gathers data from all processes onto each process.
 *
 * This overload assumes that the receive sizes are not known prior to
 * the gather. As a first step, this function gathers the number of elements
 * on all processors to the root prior to then gathering the actual elements.
 *
 * @see mxx::allgatherv()
 *
 * This function returns a `std::vector` of size
 * \f$ \sum_{i=0}^{p-1} \texttt{size@rank(i)} \f$.
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The gathered data as a `std::vector`.
 */
template <typename T>
std::vector<T> allgatherv(const std::vector<T>& data, const mxx::comm& comm = mxx::comm()) {
    return allgatherv(&data[0], data.size(), comm);
}


/*********************************************************************
 *                            All-to-all                             *
 *********************************************************************/

namespace impl {

/**
 * @brief   `all2all()` implementation for message sizes larger than MAX_INT.
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param size      The number of elements to send to each process. The `msgs` pointer
 *                  must contain `p*size` elements.
 * @param out       Pointer to memory for writing the received messages. Must be
 *                  able to hold at least `p*size` many elements of type `T`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void all2all_big(const T* msgs, size_t size, T* out, const mxx::comm& comm = mxx::comm()) {
    datatype_contiguous<T> bigtype(size);
    MPI_Alltoall(const_cast<T*>(msgs), 1, bigtype.type(), out, 1, bigtype.type(), comm);
}
}

/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * Each process contains `p = comm.size()` messages, one for each process, and
 * each containing `size` many elements of type `T`. This function sends each
 * message (the `size` many elements) to its designated target process.
 * Elements and messages are read and written from and into continuous memory.
 *
 * @see MPI_Alltoall
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param size      The number of elements to send to each process. The `msgs` pointer
 *                  must contain `p*size` elements.
 * @param out       Pointer to memory for writing the received messages. Must be
 *                  able to hold at least `p*size` many elements of type `T`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void all2all(const T* msgs, size_t size, T* out, const mxx::comm& comm = mxx::comm()) {
    if (size*comm.size() >= mxx::max_int) {
        impl::all2all_big(msgs, size, out, comm);
    } else {
        mxx::datatype<T> dt;
        MPI_Alltoall(const_cast<T*>(msgs), size, dt.type(), out, size, dt.type(), comm);
    }
}

/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * This overload returns a `size*p` size `std::vector` with the received
 * elements instead of writing output to given memory.
 *
 * @see mxx::all2all
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param size      The number of elements to send to each process. The `msgs` pointer
 *                  must contain `p*size` elements.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The received messages as a `std::vector`.
 */
template <typename T>
std::vector<T> all2all(const T* msgs, size_t size, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result(size*comm.size());
    all2all(msgs, size, &result[0], comm);
    return result;
}

/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * This overload takes a `std::vector` `msgs` for the messages to be send out.
 * The size of `msgs` must be a multiple of the number of processes in the
 * communicator `p`. This function then sends `msgs.size() / p` elements
 * to each process.
 *
 * This overload returns a `size*p` size `std::vector` with the received
 * elements instead of writing output to given memory.
 *
 * @see mxx::all2all
 *
 * @tparam T        The data type of the elements.
 * @param msgs      A `std::vector` containing elements for each process.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     the received messages as a `std::vector`.
 */
template <typename T>
std::vector<T> all2all(const std::vector<T>& msgs, const mxx::comm& comm = mxx::comm()) {
    assert(msgs.size() % comm.size() == 0);
    size_t size = msgs.size() / comm.size();
    std::vector<T> result = all2all(&msgs[0], size, comm);
    return result;
}


/*********************************************************************
 *                           All-to-all-V                            *
 *********************************************************************/

namespace impl {
/**
 * @brief   Implementation for `mxx::all2allv()` for messages sizes larger than MAX_INT
 *
 * @see mxx::all2allv()
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param send_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be send to each of the processes.
 * @param out       Pointer to memory for writing the received messages.
 * @param recv_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be received from each process.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void all2allv_big(const T* msgs, const std::vector<size_t>& send_sizes, T* out, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    // point-to-point implementation
    // TODO: implement MPI_Alltoallw variant
    // TODO: try RMA
    assert(send_sizes.size() == comm.size());
    assert(recv_sizes.size() == comm.size());
    std::vector<size_t> send_displs = impl::get_displacements(send_sizes);
    std::vector<size_t> recv_displs = impl::get_displacements(recv_sizes);
    // TODO: unify tag usage
    int tag = 12345;
    // implementing this using point-to-point communication!
    // dispatch receives
    mxx::requests reqs;
    for (int i = 0; i < comm.size(); ++i) {
        // start with self send/recv
        int recv_from = (comm.rank() + (comm.size()-i)) % comm.size();
        datatype_contiguous<T> bigtype(recv_sizes[recv_from]);
        MPI_Irecv(const_cast<T*>(&(*out)) + recv_displs[recv_from], 1, bigtype.type(),
                  recv_from, tag, comm, &reqs.add());
    }
    // dispatch sends
    for (int i = 0; i < comm.size(); ++i) {
        int send_to = (comm.rank() + i) % comm.size();
        datatype_contiguous<T> bigtype(send_sizes[send_to]);
        MPI_Isend(const_cast<T*>(msgs)+send_displs[send_to], 1, bigtype.type(), send_to,
                  tag, comm, &reqs.add());
    }

    reqs.wait();
}
}

/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * This function if for the case that the number of elements send to and received
 * from each process is not always the same, such that `all2all()` can not
 * be used.
 *
 * The number of elements send to each process from this process are given
 * by the parameter `send_sizes`.
 * The number of elements received from each process to this process are given
 * by the parameter `recv_sizes`.
 *
 * @see MPI_Alltoallv
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param send_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be send to each of the processes.
 * @param out       Pointer to memory for writing the received messages.
 * @param recv_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be received from each process.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void all2allv(const T* msgs, const std::vector<size_t>& send_sizes, T* out, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    size_t total_send_size = std::accumulate(send_sizes.begin(), send_sizes.end(), 0);
    size_t total_recv_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    size_t local_max_size = std::max(total_send_size, total_recv_size);
    mxx::datatype<size_t> mpi_sizet;
    size_t max;
    MPI_Allreduce(&local_max_size, &max, 1, mpi_sizet.type(), MPI_MAX, comm);
    if (max >= mxx::max_int) {
        impl::all2allv_big(msgs, send_sizes, out, recv_sizes, comm);
    } else {
        // convert vectors to integer counts
        std::vector<int> send_counts(send_sizes.begin(), send_sizes.end());
        std::vector<int> recv_counts(recv_sizes.begin(), recv_sizes.end());
        // get displacements
        std::vector<int> send_displs = impl::get_displacements(send_counts);
        std::vector<int> recv_displs = impl::get_displacements(recv_counts);
        // call regular alltoallv
        mxx::datatype<T> dt;
        MPI_Alltoallv(const_cast<T*>(msgs), &send_counts[0], &send_displs[0], dt.type(),
                      out, &recv_counts[0], &recv_displs[0], dt.type(), comm);
    }
}

/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * This overload returns a `std::vector` with the received data, instead of
 * writing into an output pointer `out`.
 *
 * @see all2allv()
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param send_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be send to each of the processes.
 * @param recv_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be received from each process.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The received messages as a `std::vector`.
 */
template <typename T>
std::vector<T> all2allv(const T* msgs, const std::vector<size_t>& send_sizes, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    size_t recv_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    std::vector<T> result(recv_size);
    all2allv(msgs, send_sizes, &result[0], recv_sizes, comm);
    return result;
}

/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * This overload takes a `std::vector` as input and returns
 *  a `std::vector` with the received data, instead of
 * writing into an output pointer `out`.
 *
 * @see all2allv()
 *
 * @tparam T        The data type of the elements.
 * @param msgs      `std::vector` containing the elements to be send.
 * @param send_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be send to each of the processes.
 * @param recv_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be received from each process.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The received messages as a `std::vector`.
 */
template <typename T>
std::vector<T> all2allv(const std::vector<T>& msgs, const std::vector<size_t>& send_sizes, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    assert(msgs.size() == std::accumulate(send_sizes.begin(), send_sizes.end(), 0));
    return all2allv(&msgs[0], send_sizes, recv_sizes, comm);
}

// unknown receive size:
/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * This function assumes that the `recv_sizes` are not known prior to executing
 * an `all2allv()`. Hence, this function first communicates the number of elements
 * send to each process, prior to executing a `all2allv()`.
 *
 * This overload takes a `std::vector` as input.
 *
 * @see all2allv()
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param send_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be send to each of the processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The received messages as a `std::vector`.
 */
template <typename T>
std::vector<T> all2allv(const T* msgs, const std::vector<size_t>& send_sizes, const mxx::comm& comm = mxx::comm()) {
    // first get recv sizes
    assert(send_sizes.size() == comm.size());
    std::vector<size_t> recv_sizes = all2all(send_sizes, comm);
    return all2allv(msgs, send_sizes, recv_sizes, comm);
}

/**
 * @brief   All-to-all message exchange between all processes in the communicator.
 *
 * This function assumes that the `recv_sizes` are not known prior to executing
 * an `all2allv()`. Hence, this function first communicates the number of elements
 * send to each process, prior to executing a `all2allv()`.
 *
 * This overload takes a `std::vector` as input.
 *
 * @see all2allv()
 *
 * @tparam T        The data type of the elements.
 * @param msgs      `std::vector` containing the elements to be send.
 * @param send_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be send to each of the processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 *
 * @returns     The received messages as a `std::vector`.
 */
template <typename T>
std::vector<T> all2allv(const std::vector<T>& msgs, const std::vector<size_t>& send_sizes, const mxx::comm& comm = mxx::comm()) {
    return all2allv(&msgs[0], send_sizes, comm);
}


} // namespace mxx

// include comm definitions
#define MXX_COLLECTIVE_DONE
#include "comm_def.hpp"

#endif // MXX_COLLECTIVE_HPP
