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
#include <memory>

// mxx includes
#include "datatypes.hpp"
#include "comm.hpp"


namespace mxx
{

template <typename T>
class future;

template <typename T>
class future_builder;


// TODO: put somewhere else
class requests {
public:
    requests() : m_requests() {}
    requests(MPI_Request req) : m_requests(1, req) {}
    requests(const requests& req) = default;
    requests(requests&& req) = default;
    requests& operator=(const requests& req) = default;
    requests& operator=(requests&& req) = default;

    /*
    const MPI_Request& get() const {
        return _mpi_req;
    }
    */
    void append(MPI_Request req) {
        m_requests.push_back(req);
    }

    MPI_Request& add() {
        m_requests.push_back(MPI_Request());
        return m_requests.back();
    }

    /*
    MPI_Request& get() {
        return _mpi_req;
    }
    */
    MPI_Request& back() {
        return m_requests.back();
    }

    const MPI_Request& back() const {
        return m_requests.back();
    }

    MPI_Request& operator[](std::size_t i){
        return m_requests[i];
    }

    const MPI_Request& operator[](std::size_t i) const {
        return m_requests[i];
    }

    void wait() {
        MPI_Waitall(m_requests.size(), &m_requests[0], MPI_STATUSES_IGNORE);
    }

    bool test() {
        int flag;
        MPI_Testall(m_requests.size(), &m_requests[0], &flag, MPI_STATUSES_IGNORE);
        return flag != 0;
    }

    // TODO: functions to access/return `MPI_Status`

    virtual ~requests() {
        for (MPI_Request& r : m_requests) {
            if (r != MPI_REQUEST_NULL)
                MPI_Request_free(&r);
        }
    }

private:
    std::vector<MPI_Request> m_requests;
};

/// Combines MPI request and received data storage similar to std::future
/// Calling .get() will first MPI_Wait and then std::move the data out of
/// the mxx::future
namespace impl {
template <typename T>
class future_base {
public:
    /// wrapped type
    typedef typename std::remove_reference<T>::type value_type;

    // disable copying
    future_base(const future_base& f) = delete;
    future_base& operator=(const future_base& f) = delete;

    // default move construction and assignment
    future_base(future_base&& f) = default;
    future_base& operator=(future_base&& f) = default;

    /// Returns `true` if the result is available
    bool valid() {
        if (!m_valid)
            m_valid = m_req.test();
        return m_valid;
    }

    /// blocks until the result becomes available
    void wait() {
        if (!m_valid)
            m_req.wait();
        m_valid = true;
        m_ever_valid = true;
    }

    // TODO: template specialize for <void>
    value_type get() {
        this->wait();
        m_valid = false;
        return std::move(*m_data);
    }

    virtual ~future_base() {
        // check if this has ever been valid. If not: wait(), otherwise
        // destruct the m_data member (happens anyway)
        if (!m_ever_valid && !m_valid)
            wait();
    }

protected:
    // functions only accessible by the future_builder
    value_type* data() {
        return m_data.get();
    }

    /// Default construction creates the output memory space (for MPI to write
    /// into).
    /// Only for friends!
    future_base() : m_valid(false), m_ever_valid(false) {}

    MPI_Request& add_request() {
        return m_req.add();
    }

    friend class mxx::future_builder<T>;

protected:
    typedef std::unique_ptr<value_type> ptr_type;
    ptr_type m_data;
    bool m_valid;
    bool m_ever_valid;
    requests m_req;
};

}

template <typename T>
class future : public impl::future_base<T> {
public:
    /// wrapped type
    typedef typename std::remove_reference<T>::type value_type;

    // disable copying
    future(const future& f) = delete;
    future& operator=(const future& f) = delete;

    // default move construction and assignment
    future(future&& f) = default;
    future& operator=(future&& f) = default;

    value_type get() {
        this->wait();
        this->m_valid = false;
        return std::move(*m_data);
    }

protected:
    // functions only accessible by the future_builder
    value_type* data() {
        return m_data.get();
    }

    /// Default construction creates the output memory space (for MPI to write
    /// into).
    /// Only for friends!
    future() : impl::future_base<T>(), m_data(new T()) {}

    friend class mxx::future_builder<T>;

protected:
    typedef std::unique_ptr<value_type> ptr_type;
    ptr_type m_data;
};


// template specialization for <void>
template <>
class future<void> : public impl::future_base<void> {
public:
    /// wrapped type
    typedef void value_type;

    // disable copying
    future(const future& f) = delete;
    future& operator=(const future& f) = delete;

    // default move construction and assignment
    future(future&& f) = default;
    future& operator=(future&& f) = default;

    void get() {
        wait();
        this->m_valid = false;
    }

protected:
    // functions only accessible by the future_builder
    void* data() {
        return nullptr;
    }

    /// Default construction creates the output memory space (for MPI to write
    /// into).
    /// Only for friends!
    future() : impl::future_base<void>() {}

    // declare `builder` as friend
    friend class mxx::future_builder<void>;
};


template <typename T>
class future_builder {
public:
    typedef typename  mxx::future<T>::value_type value_type;

    future_builder() : m_valid(true), m_future() {}

    MPI_Request& add_request() {
        return m_future.add_request();
    }

    value_type* data() {
        return m_future.data();
    }

    mxx::future<T> get_future() {
        m_valid = false;
        return std::move(m_future);
    }

private:
    bool m_valid;
    mxx::future<T> m_future;
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
    if (rank > 0) {
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
mxx::future<T> async_right_shift(const T& x, const mxx::comm& comm = mxx::comm()) {
    // get datatype
    mxx::datatype<T> dt;

    // TODO: handle tags with MXX (get unique tag function)
    int tag = 15;

    mxx::future_builder<T> f;
    // if not first processor
    if (comm.rank() > 0) {
        MPI_Irecv(f.data(), 1, dt.type(), comm.rank()-1, tag,
                  comm, &f.add_request());
    }
    // if not last processor
    if (comm.rank() < comm.size()-1){
        // send my most right element to the right
        MPI_Isend(const_cast<T*>(&x), 1, dt.type(), comm.rank()+1,
                  tag, comm, &f.add_request());
    }

    return std::move(f.get_future());
}

template <typename T>
mxx::future<T> async_left_shift(const T& x, const mxx::comm& comm = mxx::comm()) {
    // get datatype
    datatype<T> dt;

    // TODO: handle tags with MXX (get unique tag function)
    int tag = 15;

    mxx::future_builder<T> f;
    // if not last processor
    if (comm.rank() < comm.size()-1) {
        MPI_Irecv(f.data(), 1, dt.type(), comm.rank()+1, tag,
                  comm, &f.add_request());
    }
    // if not first processor
    if (comm.rank() > 0) {
        // send my most right element to the right
        MPI_Isend(const_cast<T*>(&x), 1, dt.type(), comm.rank()-1,
                  tag, comm, &f.add_request());
    }
    return std::move(f.get_future());
}

} // namespace mxx



#endif // MXX_SHIFT_HPP
