/**
 * @file    datatypes.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   MPI Datatypes for C++ types.
 *
 * Copyright (c) TODO
 *
 * TODO add Licence
 */

#ifndef MXX_DATATYPES_HPP
#define MXX_DATATYPES_HPP

// MPI include
#include <mpi.h>

// C++ includes
#include <vector>
#include <map>
#include <array>
#include <tuple>
#include <numeric>
#include <limits>
#include <type_traits>


namespace mxx
{

/*
 * Mapping of C/C++ types to MPI datatypes.
 *
 * Possible ways of implementation
 * 1) templated function get_mpi_dt<template T>(); (e.g. boost)
 *     -> doesn't allow partial template specialization
 * 2) overloaded functions with type deduction
 *     -> can't properly clean up types
 * 3) templated class with static method (-> allows partial specialization)
 * 4) templated class with member method (allows proper Type_free)!!!
 */

// TODO:
// - [ ] base class with MPI type modifiers (contiguous, vector, indexed, etc)
// - [x] compile time checker for builtin types
// - [ ] static function to return MPI_Datatype for builtin types!
// - [ ] put attr_map in here!
// - [ ] add to-string and caching
// - [ ] unpacking types?
// - [ ] (see wrappers in official MPI C++ bindings)

template <typename T>
class datatype {};

template <typename T>
class is_builtin_type : public std::false_type {};


// TODO: use this as a base class for datatypes?
class datatype_base {
public:
    datatype_base() : mpitype(MPI_DATATYPE_NULL), builtin(true) {
    }

    // copy constructor
    datatype_base(const datatype_base& o) {
        builtin = o.builtin;
        if (builtin) {
            mpitype = o.mpitype;
        } else {
            MPI_Type_dup(o.mpitype, &mpitype);
            MPI_Type_commit(&mpitype);
        }
    }

    // move constructor
    datatype_base(datatype_base&& o) {
        builtin = o.builtin;
        mpitype = o.mpitype;
        o.mpitype = MPI_DATATYPE_NULL;
        o.builtin = true;
    }

    // copy assignment
    datatype_base& operator=(const datatype_base& o) {
        builtin = o.builtin;
        if (builtin) {
            mpitype = o.mpitype;
        } else {
            MPI_Type_dup(o.mpitype, &mpitype);
            MPI_Type_commit(&mpitype);
        }
        return *this;
    }

    // move assignment
    datatype_base& operator=(datatype_base&& o) {
        builtin = o.builtin;
        mpitype = o.mpitype;
        o.mpitype = MPI_DATATYPE_NULL;
        o.builtin = true;
        return *this;
    }

    virtual ~datatype_base() {
        if (!builtin)
            MPI_Type_free(&mpitype);
    }
private:
    MPI_Datatype mpitype;
    bool builtin;
};

/*********************************************************************
 *                     Define built-in datatypes                     *
 *********************************************************************/

#define MXX_DATATYPE_MPI_BUILTIN(ctype, mpi_type)                           \
template <> class datatype<ctype> {                                         \
public:                                                                     \
    datatype() {}                                                           \
    MPI_Datatype type() const {return mpi_type;}                            \
    virtual ~datatype() {}                                                  \
};                                                                          \
                                                                            \
template <> class is_builtin_type<ctype> : public std::true_type {};        \

// char
MXX_DATATYPE_MPI_BUILTIN(char, MPI_CHAR);
MXX_DATATYPE_MPI_BUILTIN(unsigned char, MPI_UNSIGNED_CHAR);
MXX_DATATYPE_MPI_BUILTIN(signed char, MPI_SIGNED_CHAR);

// short
MXX_DATATYPE_MPI_BUILTIN(unsigned short, MPI_UNSIGNED_SHORT);
MXX_DATATYPE_MPI_BUILTIN(short, MPI_SHORT);

// int
MXX_DATATYPE_MPI_BUILTIN(unsigned int, MPI_UNSIGNED);
MXX_DATATYPE_MPI_BUILTIN(int, MPI_INT);

// long
MXX_DATATYPE_MPI_BUILTIN(unsigned long, MPI_UNSIGNED_LONG);
MXX_DATATYPE_MPI_BUILTIN(long, MPI_LONG);

// long long
MXX_DATATYPE_MPI_BUILTIN(unsigned long long, MPI_UNSIGNED_LONG_LONG);
MXX_DATATYPE_MPI_BUILTIN(long long, MPI_LONG_LONG);

// floats
MXX_DATATYPE_MPI_BUILTIN(float, MPI_FLOAT);
MXX_DATATYPE_MPI_BUILTIN(double, MPI_DOUBLE);
MXX_DATATYPE_MPI_BUILTIN(long double, MPI_LONG_DOUBLE);

#undef MXX_DATATYPE_MPI_BUILTIN

/**
 * @brief   MPI datatype mapping for std::array
 */
template <typename T, std::size_t size>
class datatype<std::array<T, size> > {
public:
    datatype() : _base_type() {
        MPI_Type_contiguous(size, _base_type.type(), &_type);
        MPI_Type_commit(&_type);
    }
    const MPI_Datatype& type() const {
        return _type;
    }
    MPI_Datatype& type() {
        return _type;
    }
    virtual ~datatype() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatype<T> _base_type;
};

/**
 * @brief   MPI datatype mapping for std::pair
 */
template <typename T1, typename T2>
class datatype<std::pair<T1, T2> > {
public:
    datatype() : _base_type1(), _base_type2() {
        int blocklen[2] = {1, 1};
        MPI_Aint displs[2] = {0,0};
        // get actual displacement (in case of padding in the structure)
        std::pair<T1, T2> p;
        MPI_Aint p_adr, t1_adr, t2_adr;
        MPI_Get_address(&p, &p_adr);
        MPI_Get_address(&p.first, &t1_adr);
        MPI_Get_address(&p.second, &t2_adr);
        displs[0] = t1_adr - p_adr;
        displs[1] = t2_adr - p_adr;

        // create type
        MPI_Datatype types[2] = {_base_type1.type(), _base_type2.type()};
        // in case elements are represented the opposite way around in
        // the pair (gcc does so), then swap them
        if (displs[0] > displs[1])
        {
            std::swap(displs[0], displs[1]);
            std::swap(types[0], types[1]);
        }
        MPI_Type_create_struct(2, blocklen, displs, types, &_type);
        // TODO: resize!
        MPI_Type_commit(&_type);
    }
    const MPI_Datatype& type() const {
        return _type;
    }
    MPI_Datatype& type() {
        return _type;
    }
    virtual ~datatype() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatype<T1> _base_type1;
    datatype<T2> _base_type2;
};


// fill in MPI types
template <std::size_t N, std::size_t I>
struct tuple_members
{
    template<class ...Types>
    static void get(std::map<MPI_Aint, MPI_Datatype>& members, std::tuple<datatype<Types>...>& datatypes)
    {
        // init tuple to get measurement offsets
        std::tuple<Types...> tuple;

        // get member displacement
        MPI_Aint t_adr, elem_adr;
        MPI_Get_address(&tuple, &t_adr);
        MPI_Get_address(&std::get<N-I>(tuple), &elem_adr);
        // byte offset from beginning of tuple
        MPI_Aint displ = elem_adr - t_adr;
        // fill in type
        MPI_Datatype mpi_dt = std::get<N-I>(datatypes).type();

        // add to map
        members[displ] = mpi_dt;

        // recursively (during compile time) call same function
        tuple_members<N,I-1>::get(members, datatypes);
    }
};

// Base case of meta-recursion
template <std::size_t N>
struct tuple_members<N, 0>
{
    template<class ...Types>
    static void get(std::map<MPI_Aint, MPI_Datatype>&, std::tuple<datatype<Types>...>&) {
    }
};

/**
 * @brief   MPI datatype mapping for std::tuple
 */
template <class ...Types>
class datatype<std::tuple<Types...> > {
private:
  typedef std::tuple<Types...> tuple_t;
  typedef std::tuple<datatype<Types>...> datatypes_tuple_t;
  static constexpr std::size_t size = std::tuple_size<tuple_t>::value;
public:
    datatype() : _base_types() {
        // fill in the block lengths to 1 each
        int blocklen[size];
        for (std::size_t i = 0; i < size; ++i)
        {
            blocklen[i] = 1;
        }

        // get the member displacement and type info for the tuple using
        // meta-recursion
        std::map<MPI_Aint, MPI_Datatype> members;
        tuple_members<size,size>::get(members, _base_types);


        // fill displacements and types according to in-memory order in tuple
        // NOTE: the in-memory order is not necessarily the same as the order
        // of types as accessed by std::get
        // For gcc the order is actually reversed!
        // Hence, we use a std::map to collect the order information prior
        // to creating the displacement and type arrays
        std::array<MPI_Aint, size> displs;
        std::array<MPI_Datatype, size> mpitypes;
        std::size_t i = 0;
        for (std::map<MPI_Aint, MPI_Datatype>::iterator it = members.begin();
             it != members.end(); ++it) {
            displs[i] = it->first;
            mpitypes[i] = it->second;
            ++i;
        }

        // create type
        MPI_Type_create_struct(size, blocklen, &displs[0], &mpitypes[0], &_type);
        MPI_Type_commit(&_type);

        // check extend with sizeof()
        // TODO: check arrays of elements to make sure that upper and lowerbound
        //       are set correctly (use MPI_Type_create_resized)
        MPI_Aint lb, extent;
        MPI_Type_get_extent(_type, &lb, &extent);
        if (extent != sizeof(tuple_t))
          // TODO: own error/assert handeling (MPI-Safe)
            throw std::runtime_error("MPI_Datatype extend does not match the sizeof the tuple");
    }

    const MPI_Datatype& type() const {
        return _type;
    }

    MPI_Datatype& type() {
        return _type;
    }

    virtual ~datatype() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatypes_tuple_t _base_types;
};


/*
 * "templates" for different kinds of data structures.
 * Inherit from these to specialize for your own type easily.
 */


/**
 * @brief   A contiguous datatype of the same base type
 */
template <typename T, std::size_t size = 0>
class datatype_contiguous {
  static_assert(size <= std::numeric_limits<int>::max(),
          "Compile time contiguous types only support sizes up to INT_MAX");
public:
    datatype_contiguous() : _base_type() {
        MPI_Type_contiguous(size, _base_type.type(), &_type);
        MPI_Type_commit(&_type);
    }
    virtual MPI_Datatype& type() {
        return _type;
    }
    virtual ~datatype_contiguous() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatype<T> _base_type;
};

/*
 * Runtime selection of size
 */
template <typename T>
class datatype_contiguous<T,0> {
public:
    datatype_contiguous(std::size_t size) : _base_type() {
        if (size <= std::numeric_limits<int>::max())
        {
            MPI_Type_contiguous(size, _base_type.type(), &_type);
            MPI_Type_commit(&_type);
        } else {
            // create custom data types of blocks and remainder
            std::size_t intmax = std::numeric_limits<int>::max();
            std::size_t nblocks = size / intmax;
            std::size_t rem = size % intmax;

            // create block and remainder data types
            datatype_contiguous<T, std::numeric_limits<int>::max()> _block;
            MPI_Datatype _blocks;
            MPI_Datatype _remainder;
            // create two contiguous types for blocks and remainder
            MPI_Type_contiguous(nblocks, _block.type(), &_blocks);
            MPI_Type_contiguous(rem, _base_type.type(), &_type);

            // create struct for the concatenation of this type
            MPI_Aint lb, extent;
            MPI_Type_get_extent(_base_type.type(), &lb, &extent);
            MPI_Aint displ = nblocks*intmax*extent;
            MPI_Aint displs[2] = {0, displ};
            int blocklen[2] = {1, 1};
            MPI_Datatype mpitypes[2] = {_blocks, _remainder};
            MPI_Type_create_struct(2, blocklen, displs, mpitypes, &_type);
            MPI_Type_commit(&_type);

            // clean up unused types
            MPI_Type_free(&_blocks);
            MPI_Type_free(&_remainder);
        }
    }
    virtual MPI_Datatype& type() {
        return _type;
    }
    virtual ~datatype_contiguous() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatype<T> _base_type;
};

} // namespace mxx



#endif // MXX_DATATYPES_HPP
