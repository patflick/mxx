/**
 * @file    shift.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   MPI shift communication patterns (exchange of boundary elements).
 *
 * Copyright (c) TODO
 *
 * TODO add Licence
 */

#ifndef MXX_SHIFT_HPP
#define MXX_SHIFT_HPP

// MPI include
#include <mpi.h>

// C++ includes
#include <vector>

// mxx includes
#include "datatypes.hpp"


namespace mxx
{

// TODO: put somewhere else
class request
{
private:
    MPI_Request _mpi_req;
public:
    request() : _mpi_req(MPI_REQUEST_NULL) {}
    request(MPI_Request req) : _mpi_req(req) {}
    request(const request& req) : _mpi_req(req._mpi_req) {}
    request& operator=(const request& req) {
        _mpi_req = req._mpi_req;
        return *this;
    }

    const MPI_Request& get() const {
        return _mpi_req;
    }

    MPI_Request& get() {
        return _mpi_req;
    }

    void wait() {
        MPI_Wait(&_mpi_req, MPI_STATUS_IGNORE);
    }

    bool test() {
        int flag;
        MPI_Test(&_mpi_req, &flag, MPI_STATUS_IGNORE);
        return flag != 0;
    }

    // TODO: functions to access/return `MPI_Status`

    virtual ~request() {}
};

template <typename T>
T right_shift(const T& t, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get datatype
    datatype<T> dt;
    MPI_Datatype mpi_dt = dt.type();

    // get communication parameters
    // TODO: mxx::comm or boost::mpi::comm
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // TODO: handle tags with MXX (get unique tag function)
    int tag = 13;

    T left_value; // result is the value that lies on the previous processor
    MPI_Request recv_req;
    if (rank > 0) // if not last processor
    {
        MPI_Irecv(&left_value, 1, mpi_dt, rank-1, tag,
                  comm, &recv_req);
    }
    if (rank < p-1) // if not first processor
    {
        // send my most right element to the right
        MPI_Send(const_cast<T*>(&t), 1, mpi_dt, rank+1, tag, comm);
    }
    if (rank > 0)
    {
        // wait for the async receive to finish
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    }
    return left_value;
}

template <typename T>
T left_shift(const T& t, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get datatype
    datatype<T> dt;
    MPI_Datatype mpi_dt = dt.type();

    // get communication parameters
    // TODO: mxx comm or boost::mpi::comm
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // TODO: handle tags with MXX (get unique tag function)
    int tag = 15;

    T right_value; // result is the value that lies on the previous processor
    MPI_Request recv_req;
    if (rank < p-1) // if not last processor
    {
        MPI_Irecv(&right_value, 1, mpi_dt, rank+1, tag,
                  comm, &recv_req);
    }
    if (rank > 0) // if not first processor
    {
        // send my most right element to the right
        MPI_Send(const_cast<T*>(&t), 1, mpi_dt, rank-1, tag, comm);
    }
    if (rank < p-1)
    {
        // wait for the async receive to finish
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    }
    return right_value;
}

template <typename T>
request i_right_shift(const T& input_element, T& output_element, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get datatype
    datatype<T> dt;
    MPI_Datatype mpi_dt = dt.type();

    // get communication parameters
    // TODO: mxx comm or boost::mpi::comm
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // TODO: handle tags with MXX (get unique tag function)
    int tag = 15;

    request req;
    if (rank > 0) // if not last processor
    {
        MPI_Irecv(&output_element, 1, mpi_dt, rank-1, tag,
                  comm, &req.get());
    }
    if (rank < p-1) // if not first processor
    {
        // send my most right element to the right
        MPI_Send(const_cast<T*>(&input_element), 1, mpi_dt, rank+1, tag, comm);
    }
    return req;
}

template <typename T>
request i_left_shift(const T& input_element, T& output_element, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get datatype
    datatype<T> dt;
    MPI_Datatype mpi_dt = dt.type();

    // get communication parameters
    // TODO: mxx comm or boost::mpi::comm
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // TODO: handle tags with MXX (get unique tag function)
    int tag = 15;

    request req;
    if (rank < p-1) // if not last processor
    {
        MPI_Irecv(&output_element, 1, mpi_dt, rank+1, tag,
                  comm, &req.get());
    }
    if (rank > 0) // if not first processor
    {
        // send my most right element to the right
        MPI_Send(const_cast<T*>(&input_element), 1, mpi_dt, rank-1, tag, comm);
    }
    return req;
}

} // namespace mxx



#endif // MXX_SHIFT_HPP
