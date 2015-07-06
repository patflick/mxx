/**
 * @file    comm.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements a wrapper for MPI_Comm.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef MXX_COMM_HPP
#define MXX_COMM_HPP

#include <mpi.h>

namespace mxx {

class comm {
public:
    /// Default constructor defaults to COMM_WORLD
    comm() : mpi_comm(MPI_COMM_WORLD) {
        MPI_Comm_size(mpi_comm, &m_size);
        MPI_Comm_rank(mpi_comm, &m_rank);
    }

    /// Taking an MPI_Comm object
    comm(const MPI_Comm& c) {
        if (!is_builtin(c))
            MPI_Comm_dup(c, &mpi_comm);
        else
            mpi_comm = c;
        MPI_Comm_size(mpi_comm, &m_size);
        MPI_Comm_rank(mpi_comm, &m_rank);
    }

    /// Copy constructor
    comm(const comm& o) : m_size(o.m_size), m_rank(o.m_rank) {
        if (!is_builtin(o.mpi_comm))
            MPI_Comm_dup(o.mpi_comm, &mpi_comm);
        else
            mpi_comm = o.mpi_comm;
    }

    /// Move constructor
    comm(comm&& o) : mpi_comm(o.mpi_comm), m_size(o.m_size), m_rank(o.m_rank) {
        o.mpi_comm = MPI_COMM_NULL;
        o.m_size = o.m_rank = 0;
    }

    /// Copy assignment
    comm& operator=(const comm& o) {
        free();
        if (is_builtin(o.mpi_comm))
            mpi_comm = o.mpi_comm;
        else
            MPI_Comm_dup(o.mpi_comm, &mpi_comm);
        m_size = o.m_size;
        m_rank = o.m_rank;
        return *this;
    }

    /// Move assignment
    comm& operator=(comm&& o) {
        free();
        mpi_comm = o.mpi_comm;
        m_size = o.m_size;
        m_rank = o.m_rank;
        o.mpi_comm = MPI_COMM_NULL;
        o.m_size = o.m_rank = 0;
        return *this;
    }

    /// Assigned a MPI_Comm object
    comm& operator=(const MPI_Comm& c) {
        free();
        if (is_builtin(c)) {
            mpi_comm = c;
        } else {
            MPI_Comm_dup(c, &mpi_comm);
            MPI_Comm_size(mpi_comm, &m_size);
            MPI_Comm_rank(mpi_comm, &m_rank);
        }
        return *this;
    }

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

private:
    void free() {
        if (!is_builtin())
            MPI_Comm_free(&mpi_comm);
    }

    bool is_builtin(const MPI_Comm& c) {
        return c == MPI_COMM_WORLD
            || c == MPI_COMM_SELF
            || c == MPI_COMM_NULL;
    }

    bool is_builtin() {
        return is_builtin(mpi_comm);
    }


private:
    MPI_Comm mpi_comm;
    int m_size;
    int m_rank;
};

} // namespace mxx


#endif // MXX_COMM_HPP
