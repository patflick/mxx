/**
 * @file    comm.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements a wrapper for MPI_Comm.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef MXX_COMM_FWD_HPP
#define MXX_COMM_FWD_HPP

#include <mpi.h>

#include <functional>
#include <limits>

#include "future.hpp"
#include "datatypes.hpp"

namespace mxx {

class comm {
public:
    /// Default constructor defaults to COMM_WORLD
    comm() : mpi_comm(MPI_COMM_WORLD) {
        MPI_Comm_size(mpi_comm, &m_size);
        MPI_Comm_rank(mpi_comm, &m_rank);
        do_free = false;
    }

    /// Taking an MPI_Comm object, user is responsible for freeing
    /// the communicator
    comm(const MPI_Comm& c) {
        mpi_comm = c;
        do_free = false;
        init_ranksize();
    }

public:
    // disable copying
    comm(const comm& o) = delete;
    // disable copy assignment
    comm& operator=(const comm& o) = delete;

    /// Move constructor
    comm(comm&& o) : mpi_comm(o.mpi_comm), m_size(o.m_size), m_rank(o.m_rank), do_free(o.do_free) {
        o.mpi_comm = MPI_COMM_NULL;
        o.m_size = o.m_rank = 0;
        o.do_free = false;
    }

    /// Move assignment
    comm& operator=(comm&& o) {
        free();
        mpi_comm = o.mpi_comm;
        m_size = o.m_size;
        m_rank = o.m_rank;
        do_free = o.do_free;
        o.mpi_comm = MPI_COMM_NULL;
        o.m_size = o.m_rank = 0;
        o.do_free = false;
        return *this;
    }

    /**
     * @brief   Explicity duplicate this communicator into a new communicator
     *          object. This is a collective call.
     *
     * @note    This is a collective operation and has to be called by all
     *          processes in this communicator.
     *
     * @return  The new, duplicated communicator.
     */
    comm&& copy() const {
        comm o;
        MPI_Comm_dup(mpi_comm, &o.mpi_comm);
        o.init_ranksize();
        o.do_free = true;
        return std::move(o);
    }

    /**
     * @brief   Splits the communicator into multiple sub-communicators, one for each color.
     *
     * @note    This is a collective operation and has to be called by all
     *          processes in this communicator.
     *
     * @return  The subcommunicator object.
     */
    comm split(int color) const {
        comm o;
        MPI_Comm_split(this->mpi_comm, color, this->rank(), &o.mpi_comm);
        o.init_ranksize();
        o.do_free = true;
        return std::move(o);
    }

    /**
     * @brief   Splits the communicator into multiple sub-communicators, one for each color,
     *          the order or ranks is assigned according to the given `key`.
     *
     * @note    This is a collective operation and has to be called by all
     *          processes in this communicator.
     *
     * @return  The subcommunicator object.
     */
    comm split(int color, int key) const {
        comm o;
        MPI_Comm_split(this->mpi_comm, color, key, &o.mpi_comm);
        o.init_ranksize();
        o.do_free = true;
        return std::move(o);
    }

    /**
     * @brief   Returns a new communicator which is the reverse of this.
     *
     * @note    This is a collective operation and has to be called by all
     *          processes in this communicator.
     *
     * @return  The reverse communicator.
     */
    comm reverse() const {
        comm o;
        MPI_Comm_split(this->mpi_comm, 0, this->size() - this->rank(), &o.mpi_comm);
        o.init_ranksize();
        o.do_free = true;
        return std::move(o);
    }

    /**
     * @brief   Splits this communicator into subcommunicators, one for each
     *          node/shared memory accessible regions.
     *
     * @note    This is a collective operation and has to be called by all
     *          processes in this communicator.
     *
     * @return  The subcommunicator object.
     */
    comm split_shared() const;

public:

    /// Implicit conversion to MPI_Comm
    operator MPI_Comm() const {
        return mpi_comm;
    }

    /// Destructor (frees the MPI_Comm object)
    virtual ~comm() {
        free();
    }

    /// Returns the size of the communicator
    int size() const {
        return m_size;
    }

    /// Returns the rank of this process in the communicator
    int rank() const {
        return m_rank;
    }

    void barrier() const {
        MPI_Barrier(this->mpi_comm);
    }

private:
    void free() {
        if (!is_builtin() && do_free) {
            MPI_Comm_free(&mpi_comm);
        }
    }

    bool is_builtin(const MPI_Comm& c) {
        return c == MPI_COMM_WORLD
            || c == MPI_COMM_SELF
            || c == MPI_COMM_NULL;
    }

    bool is_builtin() {
        return is_builtin(mpi_comm);
    }

    // initiate the rank and size based on the communicator
    void init_ranksize() {
        MPI_Comm_size(mpi_comm, &m_size);
        MPI_Comm_rank(mpi_comm, &m_rank);
    }

private:
    MPI_Comm mpi_comm;
    int m_size;
    int m_rank;
    bool do_free;
};



template <typename T>
mxx::future<void> async_send(const T& msg, int dest, int tag) {
    mxx::future_builder<void> f;
    mxx::datatype<T> dt;
    MPI_Isend(const_cast<T*>(&msg), 1, dt.type(), dest,
              tag, mxx::comm(), &f.add_request());
    return std::move(f.get_future());
}

template <typename T>
/// TODO: handle case if vector size is larger than INT_MAX
mxx::future<void> async_send(const std::vector<T>& msg, int dest, int tag) {
    mxx::future_builder<void> f;
    mxx::datatype<T> dt;
    MPI_Isend(const_cast<T*>(&msg[0]), msg.size(), dt.type(), dest,
              tag, mxx::comm(), &f.add_request());
    return std::move(f.get_future());
}

template <typename T>
void send(const T& msg, int dest, int tag) {

}

template <typename T>
mxx::future<T> irecv(int src, int tag) {

}

template <typename T>
struct recv_impl {
    static void do_recv_into(int src, int tag, T& buf) {
        mxx::datatype<T> dt;
        MPI_Recv(&buf, 1, dt.type(), src, tag, mxx::comm(), MPI_STATUS_IGNORE);
    }
    static T do_recv(int src, int tag) {
        T result;
        do_recv_into(src, tag, result);
        return result;
    }
};

// template specialize for std::vector (variable sized!)
template <typename T>
struct recv_impl<std::vector<T> > {
    static void do_recv_into(int src, int tag, std::vector<T>& buf) {
        // TODO: threadsafe with MProbe and MRecv (only if MPI-3)
        // TODO: how do I do this in async?
        mxx::datatype<T> dt;
        MPI_Status stat;
        MPI_Probe(src, tag, mxx::comm(), &stat);
        int count;
        MPI_Get_count(&stat, dt.type(), &count);
        std::cout << "receiving vector<int> of size: " << count << std::endl;
        if (buf.size() < count)
            buf.resize(count);
        MPI_Recv(&buf[0], count, dt.type(), stat.MPI_SOURCE, stat.MPI_TAG, mxx::comm(), MPI_STATUS_IGNORE);
    }
    static std::vector<T> do_recv(int src, int tag) {
        std::vector<T> result;
        do_recv_into(src, tag, result);
        return result;
    }
};

template <typename T>
T recv(int src, int tag) {
    return recv_impl<T>::do_recv(src, tag);
}

template <typename T>
void recv_into(int src, int tag, T& result) {
    return recv_impl<T>::do_recv_into(src, tag, result);
}
} // namespace mxx


#endif // MXX_COMM_HPP
