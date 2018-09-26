

#ifndef MXX_RMA_HPP
#define MXX_RMA_HPP

#include "common.hpp"
#include "datatypes.hpp"
#include "comm_fwd.hpp"
#include "future.hpp"


namespace mxx
{


template <typename T>
class win_target;

template <typename T>
class win {
protected:
    typedef T value_type;
    //size_t global_size;
    const mxx::comm& comm;
    mxx::datatype dt;

    MPI_Win mpi_win;
    value_type* m_data;
    size_t m_size;


    // MPI_Win_create info:
    // no_locks
    // accumulate ordering, accumulate_ops
    // same_size
    // same_disp_unit
    //
    win(const mxx::comm& comm) : comm(comm), dt(mxx::get_datatype<T>()), mpi_win(MPI_WIN_NULL), m_data(nullptr), m_size(0) {
    }
public:

    win(T* data, size_t size, const mxx::comm& comm) : comm(comm), dt(mxx::get_datatype<T>()), mpi_win(MPI_WIN_NULL), m_data(data), m_size(size) {
        // size in bytes
        // TODO: info object
        MPI_Win_create(m_data, sizeof(T)*size, sizeof(T), MPI_INFO_NULL, comm, &mpi_win);
    }


    win(size_t size, const mxx::comm& comm) : comm(comm), dt(mxx::get_datatype<T>()), mpi_win(MPI_WIN_NULL), m_data(nullptr), m_size(size) {
        // TODO: integrate info object
        MPI_Win_allocate(sizeof(T)*m_size, sizeof(T), MPI_INFO_NULL, comm, &m_data, &mpi_win);
    }

    class target_wrapper {
        win& w;
        int target_rank;

        friend class win;
        target_wrapper(win& w, int rank) : w(w), target_rank(rank) {}

    public:
        // usage examples:
        //
        // win.target(2).put(val, disp);
        // win.target(3)[13] = 5;

        // win.target(4).rput(val, disp);
        /* put */
        void put(const T* data, size_t size, size_t target_disp) {
            // TODO BigMPI for `size`
            MPI_Put(data, size, w.dt.type(), target_rank, target_disp, size, w.dt.type(), w.mpi_win);
        }
        void put(const std::vector<T>& data, size_t target_disp) {
            put(data.data, data.size(), target_disp);
        }
        void put(const T& val, size_t target_disp) {
            put(&val, 1, target_disp);
        }

        /* get */
        void get(T* origin, size_t size, size_t target_disp) const {
            // TODO BigMPI for `size` (e.g. loop over MPI_get or use dt?)
            MPI_Get(origin, size, w.dt.type(), target_rank, target_disp, size, w.dt.type(), w.mpi_win);
        }

        std::vector<T> get(size_t disp, size_t size) const {
            std::vector<T> result(size);
            get(result.data, size, disp);
            return result;
        }

        T get(size_t target_disp) const {
            T t;
            get(&t, 1, target_disp);
            return t;
        }

        mxx::request rput(const T* data, size_t size, size_t target_disp) {
            mxx::request r;
            // TODO BigMPI for `size`
            MPI_Rput(data, size, w.dt.type(), target_disp, size, w.dt.type(), w.mpi_win, r.get_ptr());
            return r;
        }
        mxx::request rput(const std::vector<T>& data, size_t target_disp) {
            return rput(data.data, data.size(), target_disp);
        }
        mxx::request rput(const T& val, size_t target_disp) {
            return rput(&val, 1, target_disp);
        }

        mxx::request rget(T* origin, size_t size, size_t target_disp) const {
            // TODO BigMPI for `size` (e.g. loop over MPI_get or use dt?)
            mxx::request r;
            MPI_Rget(origin, size, w.dt.type(), target_rank, target_disp, size, w.dt.type(), w.mpi_win, r.get_ptr());
            return r;
        }

        mxx::future<std::vector<T>> rget(size_t target_disp, size_t size) const {
            mxx::future_builder<std::vector<T>> f;
            f.data()->resize(size);
            f.add(std::move(rget(f.data().data(), size, target_disp)));
            return f.get_future();
        }

        mxx::future<T> rget(size_t target_disp) const {
            mxx::future_builder<T> f;
            f.add(std::move(rget(f.data(), 1, target_disp)));
            return f.get_future();
        }

        /* operator[] access to elements */

        class access_op {
            target_wrapper& tw;
            size_t disp;
            access_op(target_wrapper& tw, size_t disp) : tw(tw), disp(disp) {}
            friend class target_wrapper;

            public:
            inline operator T() const {
                // get value
                return tw.get(disp);
            }
            inline access_op& operator=(T val) {
                // write value
                tw.put(val, disp);
                return *this;
            }
        };

        friend class access_op;
        inline access_op operator[] (size_t disp) {
            return access_op(*this, disp);
        }

        inline const T operator[] (size_t disp) const {
            return get(disp);
        }


        /* lock/unlock passive target synchronization */

        void lock(int assert = 0) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, assert, w.mpi_win);
        }

        void lock_shared(int assert = 0) {
            MPI_Win_lock(MPI_LOCK_SHARED, target_rank, assert, w.mpi_win);
        }

        void unlock() {
            MPI_Win_unlock(target_rank, w.mpi_win);
        }

        /* flush / sync */
        void flush() {
            MPI_Win_flush(target_rank, w.mpi_win);
        }

        void flush_local() {
            MPI_Win_flush_local(target_rank, w.mpi_win);
        }
    };

    friend class target_wrapper;

    target_wrapper target(int rank) {
        return target_wrapper(*this, rank);
    }

    /// Returns a pointer to the local data
    T* data() {
        return m_data;
    }

    //
    const T* data() const {
        return m_data;
    }

    /// Returns the local number of elements in the window
    size_t size() const {
        return m_size;
    }


    /* TODO accumulate functions */


    // TODO: operator[] style access? return some object to distinguish between reading and writing
    // and only execute the necesary options
    //
    //
    // TODO: force one style of programming windows? passive vs active vs flush sync etc??

    /*
     * valid assertion flags:
     * MPI_MODE_NOCHECK, MPI_MODE_NOSTORE, MPI_MODE_NOPUT,
     * MPI_MODE_NOPRECEDE, and MPI_MODE_NOSUCCEED
     */

    // valid flags:
    // * MPI_MODE_NOSTORE, MPI_MODE_NOPUT,
    // * MPI_MODE_NOPRECEDE, and MPI_MODE_NOSUCCEED
    void fence(int assert = 0) {
        MPI_Win_fence(assert, mpi_win);
    }


    /* Exposure Epoch: (post, wait) */
    // TODO: replace MPI_Group with mxx::group

    void post(MPI_Group group, int assert = 0) {
        MPI_Win_post(group, assert, mpi_win);
    }

    void wait() {
        MPI_Win_wait(mpi_win);
    }

    bool test() {
        int flag;
        MPI_Win_test(mpi_win, &flag);
        return !(flag == 0);
    }


    /* Access Epoch: start, complete */

    void start(MPI_Group group, int assert = 0) {
        MPI_Win_start(group, assert, mpi_win);
    }

    void complete() {
        MPI_Win_complete(mpi_win);
    }


    /* Passive target synchronization (lock/unlock) */

    // TODO: create context RAII style locks (unlock on destructor)


    void lock_all(int assert = 0) {
        MPI_Win_lock_all(assert, mpi_win);
    }

    void unlock_all() {
        MPI_Win_unlock_all(mpi_win);
    }

    void flush_all() {
        MPI_Win_flush_all(mpi_win);
    }

    void flush_local_all() {
        MPI_Win_flush_local_all(mpi_win);
    }

    void sync() {
        MPI_Win_sync(mpi_win);
    }


    // TODO: get window attributes /set, get_info etc

    virtual ~win() {
        MPI_Win_free(&mpi_win);
    }
};


template <typename T>
class win_dynamic : public win<T> {
public:
    win_dynamic(const mxx::comm& comm) : win<T>(comm) {
        MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &this->mpi_win);
    }

    // TODO:
    void attach() {}
    void detach() {}

    virtual ~win_dynamic() {} // the base class already frees the win object
};


// TODO
template <typename T>
class win_shared {
};

} // namespace mxx

#endif // MXX_RMA_HPP
