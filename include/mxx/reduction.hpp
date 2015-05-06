/**
 * @file    reduction.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Reduction operations.
 *
 * TODO add Licence
 */


#ifndef MXX_REDUCTION_HPP
#define MXX_REDUCTION_HPP

#include <mpi.h>

#include <vector>
#include <iterator>
#include <limits>
#include <functional>
#include <mutex>
#include <atomic>

// mxx includes
#include "datatypes.hpp"

namespace mxx {

/*
 * Add any (key,value) pair to any MPI_Datatype using MPI caching (keyval)
 * functionality (used internally to mxx for adding std::function objects to
 * MPI datatypes, required for custom operators)
 */
template <typename K, typename T>
class attr_map {
private:
    // static "variables" wrapped into static member functions
    static int& key(){ static int k=0; return k; }
    static std::mutex& mut(){ static std::mutex m; return m; }
    static std::map<K, int>& keymap(){ static std::map<K, int> m; return m; }

    static int copy_attr(MPI_Datatype, int, void*, void *attribute_val_in, void *attribute_val_out, int *flag) {
        T** out = (T**) attribute_val_out;
        // copy via copy constructor
        *out = new T(*(T*)(attribute_val_in));
        *flag = 1;
        return MPI_SUCCESS;
    }

    static int del_attr(MPI_Datatype, int, void *attribute_val, void *) {
        delete (T*)attribute_val;
        return MPI_SUCCESS;
    }
public:
    static void set(const MPI_Datatype& dt, const K& k, const T& value) {
        std::lock_guard<std::mutex> lock(mut());
        if (keymap().find(k) == keymap().end()) {
            // create new keyval with MPI
            keymap()[k] = 0;
            MPI_Type_create_keyval(&attr_map<K,T>::copy_attr,
                                   &attr_map<K,T>::del_attr,
                                   &(keymap()[k]), (void*)NULL);
        }
        // insert new key into keymap
        // set the keyval pair
        T* val = new T(value); // copy construct
        MPI_Type_set_attr(dt, keymap()[k], (void*)val);
    }
    static T& get(const MPI_Datatype& dt, const K& k) {
        std::lock_guard<std::mutex> lock(mut());
        if (keymap().find(k) == keymap().end()) {
            throw std::out_of_range("Key not mapped.");
        }
        int key = keymap()[k];
        T* result;
        int flag;
        MPI_Type_get_attr(dt, key, &result, &flag);
        if (!flag) {
            throw std::out_of_range("Key not mapped.");
        }
        return *result;
    }
};

/*********************************************************************
 *                     User supplied functions:                      *
 *********************************************************************/

// Wraps custom operators. requires that the MPI_Datatype be duplicated
// since it attaches the operator as a std::function to the MPI_Datatype
// object.
template <typename T, bool IsCommutative = true>
class custom_op {
public:
    template <typename Func>
    custom_op(Func& func, const MPI_Datatype& type) : builtin(false) {
        // create user function
        typedef std::function<void(void*,void*,int*)> func_t;
        func_t user_func = [&func](void* invec, void* inoutvec, int* len) {
            T* in = (T*) invec;
            T* inout = (T*) inoutvec;
            for (int i = 0; i < *len; ++i) {
                inout[i] = func(in[i], inout[i]);
            }
        };
        // attach function to the datatype
        MPI_Type_dup(type, &type_copy);
        attr_map<int, func_t>::set(type_copy, 1347, user_func);

        // create op
        MPI_Op_create(&custom_op::user_function, IsCommutative, &op);
    }

    /* TODO: could potentially overload like the following for builtin
     *       functions and types. this requires checking for both: builtin type
     *       and builtin function
     *
    template <typename S>
    custom_op(std::plus<S>&, const MPI_Datatype& type)
        : builtin(true)
    {
        std::cout << "use builtin function!" << std::endl;
        type_copy = type;
        op = MPI_SUM;
    }
    */

    /**
     * @brief   Returns the MPI_Datatype which contains the std::function
     *          reference.
     */
    MPI_Datatype get_type() const {
        return type_copy;
    }

    /**
     * @brief   Returns the MPI_Op operator for reduction operations.
     */
    MPI_Op get_op() const {
        return op;
    }

    /// Destructor: cleanup MPI objects
    virtual ~custom_op() {
        if (!builtin) {
            MPI_Op_free(&op);
            MPI_Type_free(&type_copy);
        }
    }
private:
    // MPI custom Op function: (of type MPI_User_function)
    static void user_function(void* in, void* inout, int* n, MPI_Datatype* dt) {
        // get the std::function from the MPI_Datatype and call it
        typedef std::function<void(void*,void*,int*)> func_t;
        func_t f = attr_map<int, func_t>::get(*dt, 1347);
        f(in, inout, n);
    }

    /// Whether the MPI_Op is a builtin operator (e.g. MPI_SUM)
    bool builtin;
    /// The copy (Type_dup) of the MPI_Datatype to work on
    MPI_Datatype type_copy;
    /// The MPI user operator
    MPI_Op op;
};


/*********************************************************************
 *                Reductions                                         *
 *********************************************************************/
// TODO: add more (vectorized (std::vector, [begin,end),...), different reduce ops, etc)
// TODO: naming of functions !?
// TODO: come up with good naming scheme and standardize!
// TODO: template specialize for std::min, std::max, std::plus, std::multiply
//       etc for integers to use MPI builtin ops?

template <typename T>
T allreduce(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Allreduce(&x, &result, 1, dt.type(), MPI_SUM, comm);
    return result;
}

template <typename T>
T reduce(T& x, MPI_Comm comm = MPI_COMM_WORLD, int root = 0)
{
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Reduce(&x, &result, 1, dt.type(), MPI_SUM, root, comm);
    return result;
}

template <typename T, typename Func>
T allreduce(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    // get custom op (and type for custom op)
    mxx::custom_op<T> op(func, dt.type());
    // perform reduction
    T result;
    MPI_Allreduce(&x, &result, 1, op.get_type(), op.get_op(), comm);
    // return result
    return result;
}

template <typename T>
T exscan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Exscan(&x, &result, 1, dt.type(), MPI_SUM, comm);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
      result = T();
    return result;
}

template <typename T, typename Func>
T exscan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    // get op
    mxx::custom_op<T> op(func, dt.type());
    // perform reduction
    T result;
    MPI_Exscan(&x, &result, 1, op.get_type(), op.get_op(), comm);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
      result = T();
    return result;
}

template <typename T>
T scan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Scan(&x, &result, 1, dt.type(), MPI_SUM, comm);
    return result;
}

template <typename T, typename Func>
T scan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    // get op
    mxx::custom_op<T> op(func, dt.type());
    T result;
    MPI_Scan(&x, &result, 1, op.get_type(), op.get_op(), comm);
    return result;
}

/****************************************************
 *  reverse reductions (with reverse communicator)  *
 ****************************************************/

inline void rev_comm(MPI_Comm comm, MPI_Comm& rev)
{
    // get MPI parameters
    int rank;
    int p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    MPI_Comm_split(comm, 0, p - rank, &rev);
}

template <typename T>
T reverse_exscan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = exscan(x, rev);
    MPI_Comm_free(&rev);
    return result;
}

template <typename T, typename Func>
T reverse_exscan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = exscan(x, func, rev);
    MPI_Comm_free(&rev);
    return result;
}

template <typename T>
T reverse_scan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = scan(x, rev);
    MPI_Comm_free(&rev);
    return result;
}

template <typename T, typename Func>
T reverse_scan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = scan(x, func, rev);
    MPI_Comm_free(&rev);
    return result;
}


/*********************
 *  Specialized ops  *
 *********************/

/************************
 *  Boolean reductions  *
 ************************/
// useful for testing global conditions, such as termination conditions

inline bool all_of(bool x, MPI_Comm comm = MPI_COMM_WORLD) {
    int i = x ? 1 : 0;
    int result;
    MPI_Allreduce(&i, &result, 1, MPI_INT, MPI_LAND, comm);
    return result != 0;
}

inline bool any_of(bool x, MPI_Comm comm = MPI_COMM_WORLD) {
    int i = x ? 1 : 0;
    int result;
    MPI_Allreduce(&i, &result, 1, MPI_INT, MPI_LOR, comm);
    return result != 0;
}

inline bool none_of(bool x, MPI_Comm comm = MPI_COMM_WORLD) {
    int i = x ? 1 : 0;
    int result;
    MPI_Allreduce(&i, &result, 1, MPI_INT, MPI_LAND, comm);
    return result == 0;
}

} // namespace mxx

#endif // MXX_REDUCTION_HPP
