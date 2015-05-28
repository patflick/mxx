/**
 * @file    collective.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Collective operations.
 * @group   collective
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
#include "shift.hpp"
#include "comm.hpp"

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
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::requests reqs;
    int tag = 1234; // TODO: handle tags somewhere (as attributes in the comm?)
    if (comm.rank() == root) {
        mxx::datatype_contiguous<T> dt(size);
        for (int i = 0; i < comm.size(); ++i) {
            if (i == root) {
                // copy input into output
                std::copy(msgs+i*size, msgs+(i+1)*size, out);
            } else {
                MPI_Isend(const_cast<T*>(msgs)+i*size, 1, dt.type(), i, tag, comm, &reqs.add());
            }
        }
    } else {
        // create custom datatype to encapsulate the whole message
        mxx::datatype_contiguous<T> dt(size);
        MPI_Irecv(const_cast<T*>(out), 1, dt.type(), root, tag, comm, &reqs.add());
    }
    reqs.wait();
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



} // namespace mxx

#endif // MXX_COLLECTIVE_HPP
