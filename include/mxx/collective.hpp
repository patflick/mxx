/**
 * @file    collective.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Collective operations.
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
//#include <iterator>
#include <limits>
//#include <functional>

// mxx includes
#include "datatypes.hpp"
#include "shift.hpp"
//#include "algos.hpp"
//#include "partition.hpp"

#include "comm.hpp"

namespace mxx {

// TODO: unify this with regular messages
#ifdef MXX_MAX_INT
// set to smaller value for testing
constexpr size_t max_int = MXX_MAX_INT;
#else
// maximum message size for MPI
constexpr size_t max_int = std::numeric_limits<int>::max();
#endif

/*********************************************************************
 *                              Scatter                              *
 *********************************************************************/

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

template <typename T>
void scatter(const T* msgs, size_t size, T* out, int root, const mxx::comm& comm = mxx::comm())
{
    if (size*comm.size() >= mxx::max_int) {
        // own scatter for large messages
        scatter_big(msgs, size, out, root, comm);
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

template <typename T>
std::vector<T> scatter(const T* msgs, size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result(size);
    scatter(msgs, size, &result[0], root, comm);
    return std::move(result);
}

template <typename T>
std::vector<T> scatter(const std::vector<T>& msgs, size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    assert(comm.rank() != root || msgs.size() == size*comm.size());
    std::vector<T> result = scatter(&msgs[0], size, root, comm);
    return std::move(result);
}

template <typename T>
std::vector<T> scatter_recv(size_t size, int root, const mxx::comm& comm = mxx::comm()) {
    std::vector<T> result(size);
    scatter((T*)nullptr, size, &result[0], root, comm);
    return std::move(result);
}

/************************
 *  Scatter size first  *
 ************************/

template <typename T>
std::vector<T> scatter(const std::vector<T>& msgs, int root, const mxx::comm& comm = mxx::comm()) {
    size_t size = msgs.size();
    mxx::datatype<size_t> sizedt;
    MPI_Bcast(&size, 1, sizedt.type(), root, comm);
    // now everybody knows the size
    std::vector<T> result = scatter(msgs, size, root, comm);
    return std::move(result);
}

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

// alias for size = 1, returns the type instead of a vector
template <typename T>
T scatter_one(const std::vector<T>& msgs, int root, const mxx::comm& comm = mxx::comm()) {
    assert(comm.rank() != root || msgs.size() == comm.size());
    T result;
    scatter(&msgs[0], 1, &result, root, comm);
    return result;
}

template <typename T>
T scatter_one_recv(int root, const mxx::comm& comm = mxx::comm()) {
    T result;
    scatter((T*)nullptr, 1, &result, root, comm);
    return result;
}

// TODO: gtest or other test suite for testing MPI

/*********************************************************************
 *                             Scatter-V                             *
 *********************************************************************/



} // namespace mxx

#endif // MXX_COLLECTIVE_HPP
