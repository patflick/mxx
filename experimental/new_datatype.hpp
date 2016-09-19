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

template <typename T>
struct datatype_descriptor {
    //static constexpr bool has_descriptor = false;
};


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

// TODO: new datatype builder should be able to build a datatype by recursivly
//       adding all members to the same datatype (instead of MPI_Type_struct of already custom members)

// build datatype from tuple of pointer-to-member:
// build datatype from tuple of pointers

template <typename T, typename Enable = void>
struct has_type_descriptor : std::false_type {};

template <typename T>
struct has_type_descriptor<T, typename std::enable_if<has_static_member_object_datatype<T>::value>::type> : std::true_type {};

template <typename T>
struct has_type_descriptor<T, typename std::enable_if<has_static_member_object_datatype<datatype_descriptor<T>>::value>::type> : std::true_type {};

/*
template <typename T, typename Enable = void>
std::false_type constexpr get_datatype_descriptor() {
    // TODO: static assert
    return std::false_type();
}
*/

template <typename T>
auto constexpr get_datatype_descriptor(typename std::enable_if<has_static_member_object_datatype<T>::value, int>::type = 0) -> decltype(T::datatype) {
    return T::datatype;
}

template <typename T>
auto constexpr get_datatype_descriptor(typename std::enable_if<has_static_member_object_datatype<datatype_descriptor<T>>::value, long>::type = 0) -> decltype(datatype_descriptor<T>::datatype) {
    return datatype_descriptor<T>::datatype;
}

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
struct buf_instance;

// encapsulates a buffer via size and buf member
// TODO: rename as member_ptr_buffer_decl
template <typename T, typename U, typename S>
struct buffer_decl {
    typedef T class_type;
    typedef U value_type;
    typedef S size_type;
    typedef buf_instance<buffer_decl<T, U, S>> instance_type;

    // pointer to member
    S T::* m_size;
    U* T::* m_buf;

    instance_type instance_wrapper(T& t) {
       return instance_type(*this, t);
    }

    inline size_type size(const T& instance) const {
        return instance.*m_size;
    }

    inline const value_type* data(const T& instance) const {
       return instance.*m_buf;
    }

    inline value_type* data(T& instance) const {
       return instance.*m_buf;
    }

    inline void resize(T& instance, size_type size) {
       // TODO dealloc?
       instance.*m_size = size;
       instance.*m_buf = new value_type[size];
    }

    constexpr buffer_decl(S T::* size, U* T::* buf) : m_size(size), m_buf(buf) {}
    constexpr buffer_decl(const buffer_decl& o) : m_size(o.size), m_buf(o.buf) {}
};


template <typename T, typename U, typename S>
struct buf_instance<buffer_decl<T,U,S>> {
    buffer_decl<T,U,S> b;
    T& instance;

    typedef buffer_decl<T,U,S> buffer_type;
    typedef typename buffer_type::size_type size_type;
    typedef typename buffer_type::value_type value_type;

    buf_instance(buffer_decl<T,U,S>& b, T& instance) : b(b), instance(instance) {}

    // standard accessors
    size_type size() const {
        return instance.*b.size;
    }

    value_type* data() {
        return instance.*b.buf;
    }

    const value_type* data() const {
        return instance.*b.buf;
    }

    void resize(size_type size) {
        // allocate data and set size
        instance.*b.buf = new value_type[size];
        instance.*b.size = size;
    }
};


// std::vector etc accesors
template <typename C>
struct get_data_func {
    inline auto operator()(C& c) const -> decltype(std::declval<C>().data()) {
        return c.data();
    }
    inline auto operator()(const C& c) const -> decltype(std::declval<const C>().data()) {
        return c.data();
    }
};

template <typename C>
struct resize_alloc {
    template <typename SizeT>
    inline void operator()(C& c, SizeT size) const {
        c.resize(size);
    }
};

template <typename C>
struct size_member {
    inline auto size(const C& c) const -> decltype(std::declval<C>().size()) {
       return c.size();
    }
};

template <typename T, typename Alloc, typename Data, typename Size = size_member<T>>
struct custom_buffer_decl {
    typedef T class_type;
    typedef decltype(std::declval<Data>()(std::declval<T>())) pointer_type;
    typedef typename std::remove_pointer<pointer_type>::type value_type;

    Alloc alloc_functor;
    Data data_functor;
    Size size_functor;

    constexpr custom_buffer_decl(Alloc&& a, Data&& d, Size&& s) : alloc_functor(a), data_functor(d), size_functor(s) {}
    // use the default .size() accessor
    constexpr custom_buffer_decl(Alloc&& a, Data&& d) : alloc_functor(a), data_functor(d), size_functor(Size()) {}
};

template <typename B, typename Enable = void>
struct is_buffer : std::false_type {};

template <typename T, typename Alloc, typename Data, typename Size>
struct is_buffer<custom_buffer_decl<T, Alloc, Data, Size>> : std::true_type {};

template <typename T, typename U, typename S>
struct is_buffer<buffer_decl<T, U, S>> : std::true_type {};

template <typename T, typename Alloc, typename Data>
constexpr custom_buffer_decl<T,Alloc,Data> custom_buffer_tag(Alloc&& a, Data&& d) {
    return custom_buffer_decl<T, Alloc, Data>(std::forward<Alloc>(a), std::forward<Data>(d));
}

template <typename T, typename Alloc, typename Data, typename Size>
constexpr custom_buffer_decl<T,Alloc,Data,Size> custom_buffer_tag(Alloc&& a, Data&& d, Size&& s) {
    return custom_buffer_decl<T, Alloc, Data, Size>(std::forward<Alloc>(a), std::forward<Data>(d), std::forward<Size>(s));
}

/* TODO: range tag and "flattening" vs serialization of ranges */


template <typename T>
struct get_descriptor_type {
   typedef typename std::remove_cv<decltype(get_datatype_descriptor<T>())>::type type;
};

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


template <typename T, typename Alloc>
struct datatype_descriptor<std::vector<T, Alloc>> {
    constexpr static auto datatype = std::make_tuple(custom_buffer_tag<T>(resize_alloc<std::vector<T, Alloc>>(), get_data_func<std::vector<T, Alloc>>()));
};


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


template <typename T>
typename std::enable_if<has_type_descriptor<T>::value, mxx::datatype>::type
make_datatype() {
    recursive_datatype_builder<T> builder;
    builder.build();
    return builder.get_datatype();
}

// TODO: new flat_type implementation (compile time buffers, rather than the
//       shitty type-unsafe implementation we currently have)
// TODO: serialization (of std containers) implementation!


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
