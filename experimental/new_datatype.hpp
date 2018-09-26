#ifndef MXX_EXPERIMENTAL_NEW_DATATYPE
#define MXX_EXPERIMENTAL_NEW_DATATYPE

// experimental implementation of new mxx::datatype deduction

// type can be builtin, simple, flat, flat_container, or serializable

// declaration via (constexpr) static tuple of pointer to object member
// or via template specializing a constuctor class

#include <cstddef>
#include <utility>
#include <tuple>
#include <stack>

#include "type_traits.hpp"
#include "member_tuple.hpp"
#include <mxx/datatypes.hpp>

namespace mxx {

MXX_GENERATE_HAS_STATIC_MEMBER_OBJECT(datatype)

MXX_GENERATE_HAS_MEMBER_FUNCTION(size)

/**********************************************************************
*                        Datatype descriptors                        *
**********************************************************************/

template <typename T>
struct datatype_descriptor {};

template <typename T, typename U>
struct datatype_descriptor<std::pair<T, U>> {
    typedef std::pair<T, U> type;
    // define all members
    static constexpr auto datatype = std::make_tuple(&type::first, &type::second);
};

// helper function to get pointers to all tuple members
template <size_t... Seq, typename... Types>
constexpr std::tuple<Types*...> get_tuple_offsets(std::tuple<Types...>* p, seq_<Seq...>) {
    return std::tuple<Types*...>(&std::get<Seq>(*p)...);
}

template <typename... Types>
struct datatype_descriptor<std::tuple<Types...>> {
    static constexpr std::tuple<Types...>* null_tuple = nullptr;
    // declare tuple members via pointers from nullptr since we can't access
    // the members itself to get the pointer-to-member
    static constexpr auto datatype = get_tuple_offsets(null_tuple, typename seq<sizeof...(Types)>::type());
};

template <typename T, typename Enable = void>
struct has_type_descriptor : std::false_type {};

template <typename T>
struct has_type_descriptor<T, typename std::enable_if<has_static_member_object_datatype<T>::value>::type> : std::true_type {};

template <typename T>
struct has_type_descriptor<T, typename std::enable_if<has_static_member_object_datatype<datatype_descriptor<T>>::value>::type> : std::true_type {};

template <typename T>
auto constexpr get_datatype_descriptor(typename std::enable_if<has_static_member_object_datatype<T>::value, int>::type = 0) -> decltype(T::datatype) {
    return T::datatype;
}

template <typename T>
auto constexpr get_datatype_descriptor(typename std::enable_if<has_static_member_object_datatype<datatype_descriptor<T>>::value, long>::type = 0) -> decltype(datatype_descriptor<T>::datatype) {
    return datatype_descriptor<T>::datatype;
}


/**********************************************************************
*                     Recursive datatype builder                     *
**********************************************************************/

template <typename T>
struct recursive_datatype_builder;

template <typename Foo, typename U>
struct recursive_datatype_helper : public recursive_processor<recursive_datatype_helper<Foo,U>> {
private:
    recursive_datatype_builder<Foo>& builder;
    size_t offset;

    // for members of type `U`
    template <typename M>
    size_t offset_from_ptr(M* m) {
        size_t offset = reinterpret_cast<size_t>(m);
        MXX_ASSERT(0 <= offset && offset + sizeof(M) <= sizeof(U));
        return offset;
    }

    typedef recursive_processor<recursive_datatype_helper<Foo,U>> rec_process;

public:
    recursive_datatype_helper(recursive_datatype_builder<Foo>& b, size_t offset)
        : builder(b), offset(offset) {}

    // add members via "pointer to member" types
    template <typename M>
    typename std::enable_if<is_builtin_type<M>::value, void>::type
    add_member(M U::*m) {
        builder.template add_member_by_offset<M>(offset + offset_of<U, U, M>(m));
    }

    // support adding members of base classes
    template <typename Base, typename M>
    typename std::enable_if<std::is_base_of<Base, U>::value
    && is_builtin_type<M>::value, void>::type
    add_member(M Base::*m) {
        builder.template add_member_by_offset<M>(offset + offset_of<U, Base, M>(m));
    }

    // add by pointer!
    template <typename M>
    typename std::enable_if<is_builtin_type<M>::value, void>::type
    add_member(M* m) {
        builder.template add_member_by_offset<M>(offset + offset_from_ptr(m));
    }

    /*******************************************
     *  For types that have a type descriptor  *
     *******************************************/

    // add members via "pointer to member" types
    template <typename M>
    typename std::enable_if<has_type_descriptor<M>::value, void>::type
    add_member(M U::*m) {
        recursive_datatype_helper<Foo, M> h(builder, offset_of<U, U, M>(m));
        // get datatype descriptor
        auto descr = get_datatype_descriptor<M>();
        // unpack and pass to recursive helper
        h.unpack_tuple(descr, typename seq<std::tuple_size<decltype(descr)>::value>::type());
    }

    // support adding members of base classes
    template <typename Base, typename M>
    typename std::enable_if<std::is_base_of<Base, U>::value
    && has_type_descriptor<M>::value, void>::type
    add_member(M Base::*m) {
        recursive_datatype_helper<Foo, M> h(builder, offset_of<U, Base, M>(m));
        // get datatype descriptor
        auto descr = get_datatype_descriptor<M>();
        // unpack and pass to recursive helper
        h.unpack_tuple(descr, typename seq<std::tuple_size<decltype(descr)>::value>::type());
    }

    // add by pointer!
    template <typename M>
    typename std::enable_if<has_type_descriptor<M>::value, void>::type
    add_member(M* m) {
        recursive_datatype_helper<Foo, M> h(builder, offset_from_ptr(m));
        // get datatype descriptor
        auto descr = get_datatype_descriptor<M>();
        // unpack and pass to recursive helper
        h.unpack_tuple(descr, typename seq<std::tuple_size<decltype(descr)>::value>::type());
    }

    template <typename M>
    void process(M&& m) {
        this->add_member(std::forward<M>(m));
    }

    template <typename... Types>
    void add_members(Types&&...t) {
        // recursively process using the recursive_processor base class
        // which will call our `process(M)` function for each of the passed types
        rec_process::process(std::forward<Types>(t)...);
    }

    template <size_t... Seq, typename... Types>
    void unpack_tuple(const std::tuple<Types...>& t, seq_<Seq...>) {
        rec_process::process(std::get<Seq>(t)...);
    }
};

template <typename T>
struct get_descriptor_type {
   typedef typename std::remove_cv<decltype(get_datatype_descriptor<T>())>::type type;
};

/* TODO
 *  rename `simple`?
 *  Current definition:
 *  - a type T is a simple_type iff one of:
 *      - T is a builtin type
 *      - T has a datatype_descriptor and all members are simple (recursive)
 */
template <typename T>
struct is_simple_type;

template <typename T>
struct is_simple_member : std::false_type {};

template <typename T, typename M>
struct is_simple_member<M T::*> : is_simple_type<M> {};

template <typename M>
struct is_simple_member<M*> : is_simple_type<M> {};

// all builtin types are simple
template <typename T, typename Enable = void>
struct is_simple_type_helper : std::integral_constant<bool, is_builtin_type<T>::value> {};

// any type with a datatype descriptor of all simple types is simple
template <typename T>
struct is_simple_type_helper<T, typename std::enable_if<
   !is_builtin_type<T>::value
   && has_type_descriptor<T>::value>::type>
   : std::integral_constant<bool,
   all_are<is_simple_member, typename get_descriptor_type<T>::type>::value
   > {};

template <typename T>
struct is_simple_type : is_simple_type_helper<T> {};




// TODO: the API needs some work
template <typename T>
struct recursive_datatype_builder : public datatype_builder_base<T, recursive_datatype_builder<T>> {
private:
    typedef datatype_builder_base<T, recursive_datatype_builder<T>> base_type;
   // std::stack<size_t> offset_stack;
public:
    recursive_datatype_builder() : base_type() {}

    /*
    template <typename... Types>
    void add_members(Types&&...t) {
        recursive_datatype_helper<T, T> h(*this, 0);
        h.add_members(std::forward<Types>(t)...);
    }
    */

    void build() {
        recursive_datatype_helper<T, T> h(*this, 0);
        // get datatype descriptor
        auto descr = get_datatype_descriptor<T>();
        // unpack and pass to recursive helper
        h.unpack_tuple(descr, typename seq<std::tuple_size<decltype(descr)>::value>::type());
    }
};


// should be only if it recursively has the type descriptor and at the bottom each type is builtin (or trivially_copyable into a fixed number of bytes!?)
template <typename T>
typename std::enable_if<has_type_descriptor<T>::value, mxx::datatype>::type
make_datatype() {
    recursive_datatype_builder<T> builder;
    builder.build();
    return builder.get_datatype();
}




/**********************************************************************
*                     Type decoding (pictogram)                      *
**********************************************************************/

// flat type representation used for decoding
struct flat_repr {
    MPI_Datatype type;
    std::map<MPI_Aint, MPI_Datatype> m;
    MPI_Aint cur_offset;

    flat_repr() : m(), cur_offset(0) {}

    void print_pictogram(std::ostream& os, unsigned int width = 8) {
        std::pair<MPI_Aint, MPI_Aint> ext;
        std::pair<MPI_Aint, MPI_Aint> true_ext;
        MPI_Type_get_extent(type, &ext.first, &ext.second);
        MPI_Type_get_true_extent(type, &true_ext.first, &true_ext.second);

        if (ext.first != 0) {
            os << "Pictogram not available for types with lb != 0" << std::endl;
            return;
        }

        MPI_Aint ex = ext.second;

        if (ext.second < true_ext.second) {
            os << "Pictogram not available for types with ub < true_ub" << std::endl;
            return;
        }

        // use single letter type
        unsigned int pos = 0;
        os << "[";
        for (unsigned int i = 0; i < width; ++i) {
            os << "-";
        }
        os << "]" << std::endl << "[";
        for (auto it = m.begin(); it != m.end(); ++it) {
            while (pos < it->first) {
                if (pos % width == 0 && pos != 0)
                    os << "]" << std::endl << "[";
                os << " ";
                ++pos;
            }
            // get type size and type char
            char type_char = builtin_typename_map::get_typeid_name(it->second)[0];
            int size;
            MPI_Type_size(it->second, &size);
            // print the character `size` times
            for (int i = 0; i < size; ++i) {
                if (pos % width == 0 && pos != 0)
                    os << "]" << std::endl << "[";
                os << type_char;
                ++pos;
            }
        }

        while (pos < ex) {
            if (pos % width == 0 && pos != 0)
                os << "]" << std::endl << "[";
            os << " ";
            ++pos;
        }

        os << "]" << std::endl << "[";
        for (unsigned int i = 0; i < width; ++i) {
            os << "-";
        }
        os << "]" << std::endl;
    }

    void print_pictogram(unsigned int width = 8) {
        print_pictogram(std::cout, width);
    }
};



struct type_decoder {
    static void unpack_envelope(MPI_Datatype type, flat_repr& f) {
        int num_ints, num_addr, num_dt, comb;
        MPI_Type_get_envelope(type, &num_ints, &num_addr, &num_dt, &comb);

        if (comb == MPI_COMBINER_NAMED) {
            //std::cout << "Type: " << builtin_typename_map::get_typeid_name(type) << std::endl;
            f.m.emplace(f.cur_offset, type);
            return;
        }

        // allocate the output for get_contents
        std::vector<int> ints; ints.resize(num_ints);
        std::vector<MPI_Aint> addrs; addrs.resize(num_addr);
        std::vector<MPI_Datatype> types; types.resize(num_dt);

        MPI_Type_get_contents(type, num_ints, num_addr, num_dt,
                              &ints[0], &addrs[0], &types[0]);

        switch(comb) {
          case MPI_COMBINER_DUP:
            MXX_ASSERT(num_ints == 0 && num_addr == 0 && num_dt == 1);
            unpack_envelope(types[0], f);
            break;
          case MPI_COMBINER_CONTIGUOUS:
            std::cout << "Contiguous: " << ints[0] << " x ";
            unpack_envelope(types[0], f);
            break;
          case MPI_COMBINER_VECTOR:
          case MPI_COMBINER_HVECTOR:
          case MPI_COMBINER_INDEXED:
          case MPI_COMBINER_HINDEXED:
          case MPI_COMBINER_INDEXED_BLOCK:
          case MPI_COMBINER_HINDEXED_BLOCK:
            std::cout << "NOT YET SUPPORTED vector/indexed/indexed_block" << std::endl;
            break;
          case MPI_COMBINER_STRUCT:
            {
                int count = ints[0];
                std::vector<int> blen(&ints[1], &ints[0]+count);
                std::vector<MPI_Aint> displ = addrs;
                std::cout << "Struct: " << std::endl;
                MPI_Aint offset = f.cur_offset;
                for (int i = 0; i < count; ++i) {
                    f.cur_offset = offset + displ[i];
                    unpack_envelope(types[i], f);
                }
                f.cur_offset = offset;
            }
            break;
          case MPI_COMBINER_RESIZED:
            // TODO
            std::cout << "resized to [" << addrs[0] << "," << addrs[1] << "): " << std::endl;
            unpack_envelope(types[0], f);
            break;
          case MPI_COMBINER_SUBARRAY:
          case MPI_COMBINER_DARRAY:
            std::cout << "NOT YET SUPPORTED subarray/darray" << std::endl;
            break;
        }
    }
};

} // namespace mxx

#endif // MXX_EXPERIMENTAL_NEW_DATATYPE
