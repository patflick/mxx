#ifndef MXX_EXPERIMENTAL_BUFFER
#define MXX_EXPERIMENTAL_BUFFER

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
/**********************************************************************
*                            Buffer stuff                            *
**********************************************************************/

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

template <typename B>
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

template <typename T, typename Alloc>
struct datatype_descriptor<std::vector<T, Alloc>> {
    constexpr static auto datatype = std::make_tuple(custom_buffer_tag<T>(resize_alloc<std::vector<T, Alloc>>(), get_data_func<std::vector<T, Alloc>>()));
};

/**********************************************************************
*                           Template magic                           *
**********************************************************************/
// TODO
// - what is all this stuff used for??
// - what did I plan this stuff for? has to be due to std::vector stuff, right?

template <typename... Types>
struct types {
    static constexpr size_t size = sizeof...(Types);
    using tuple_type = std::tuple<Types...>;
};

template <typename, typename...>
struct types_insert;

template <typename T, typename... Types>
struct types_insert<T, types<Types...>> {
    using type = types<T, Types...>;
};

template <template<typename> class type_trait, typename... Types>
struct type_filter;

template <template<typename> class type_trait>
struct type_filter<type_trait> {
    using type = types<>;
};

template <template<typename> class type_trait, typename Type, typename... Types>
struct type_filter<type_trait, Type, Types...> {
    using type = typename std::conditional<type_trait<Type>::value,
          typename types_insert<Type, typename type_filter<type_trait, Types...>::type>::type,
          typename type_filter<type_trait, Types...>::type>::type;
};

// unpack `types`
template <template<typename> class type_trait, typename... Types>
struct type_filter<type_trait, types<Types...>> : type_filter<type_trait, Types...> {};

template <typename...>
struct types_from_tuple;

template <typename... Types>
struct types_from_tuple<std::tuple<Types...>> {
    using type = types<Types...>;
};

template <typename T>
struct get_buffer_types {
    // TODO: can't use T::datatype, since it may be type_declarator<T>::datatype
    // TODO: use struct to combine those
    using type = typename type_filter<is_buffer, typename types_from_tuple<decltype(T::datatype)>::type>::type;
};

template <template<typename> class type_trait, typename Types, size_t I = 0>
struct indeces_of;

template <template<typename> class type_trait, size_t I>
struct indeces_of<type_trait, types<>, I> {
    using seq = seq_<>;
};

template <size_t I, typename seq>
struct insert_num;

template <size_t I, size_t... Seq>
struct insert_num<I, seq_<Seq...>> {
    using seq = seq_<I, Seq...>;
};

template <template<typename> class type_trait, size_t I, typename T, typename... Types>
struct indeces_of<type_trait, types<T, Types...>, I> {
    using seq = typename std::conditional<type_trait<T>::value,
          typename insert_num<I, typename indeces_of<type_trait, types<Types...>, I+1>::seq>::seq,
          typename indeces_of<type_trait, types<Types...>, I+1>::seq>::type;
};

// example for a caller function, which calls the size() function
struct size_caller {
    template <typename T, typename... Args>
    auto operator()(T&& t, Args&&... args) -> decltype(t.size(std::forward<Args>(args)...)) {
        return t.size(std::forward<Args>(args)...);
    }
};

// TODO helper with unpack tuple arguments
template <typename T, size_t I>
struct unpack_if_tuple {
   static T&& unpack(T&& t) {
       return std::forward<T>(t);
   }
};

// specialize for tuple
template <size_t I, typename... Types>
struct unpack_if_tuple<std::tuple<Types...>, I> {
   using return_type = typename std::tuple_element<I, std::tuple<Types...>>::type;
   static return_type&& unpack(std::tuple<Types...>& t) {
       return std::forward<return_type>(std::get<I>(t));
   }
   static const return_type&& unpack(const std::tuple<Types...>& t) {
       return std::forward<const return_type>(std::get<I>(t));
   }
};

template <size_t I, size_t J, typename Caller, typename Tuple, typename... Args>
auto unpack_caller(Tuple&& t, Caller&& caller, Args&&... args)
   -> decltype(caller(std::get<I>(t), unpack_if_tuple<Args, J>::unpack(args)...)) {
    return caller(std::get<I>(t), unpack_if_tuple<Args, J>::unpack(args)...);
}

//// TODO: collapse void
//template <typename Caller, typename Tuple, typename... Args>
//struct unpack_return_type;
//
//template <typename Caller, typename... Types, typename... Args>
//struct unpack_return_type<Caller, std::tuple<Types...>, Args...>
//{
//    using return_type = std::tuple<decltype(unpack_caller<Seq, Seq2>(std::forward<Tuple>(t), std::forward<Caller>(caller), std::forward<Args>(args)...))...>;
//};

template <typename Caller, size_t... Seq, size_t... Seq2, typename Tuple, typename... Args>
auto call_all_unpack_helper(seq_<Seq...>, seq_<Seq2...>, Tuple&& t, Caller&& caller, Args&&... args)
   -> std::tuple<decltype(unpack_caller<Seq, Seq2>(std::forward<Tuple>(t), std::forward<Caller>(caller), std::forward<Args>(args)...))...> {
    /// return type is tuple of caller applied to each type in Types with args
    using return_type = std::tuple<decltype(unpack_caller<Seq, Seq2>(std::forward<Tuple>(t), std::forward<Caller>(caller), std::forward<Args>(args)...))...>;
    return return_type(unpack_caller<Seq, Seq2>(std::forward<Tuple>(t), std::forward<Caller>(caller), std::forward<Args>(args)...)...);
};

template <typename Caller, typename... Types, typename... Args>
auto call_all_unpack(std::tuple<Types...>& t, Caller&& caller, Args&&... args)
   -> decltype(call_all_unpack_helper(typename seq<sizeof...(Types)>::type(), typename seq<sizeof...(Types)>::type(), t, std::forward<Caller>(caller), std::forward<Args>(args)...))
{
   return call_all_unpack_helper(typename seq<sizeof...(Types)>::type(), typename seq<sizeof...(Types)>::type(), t, std::forward<Caller>(caller), std::forward<Args>(args)...);
}


template <typename Caller, size_t... Seq, typename Tuple, typename... Args>
auto call_all_helper(seq_<Seq...>, Tuple&& t, Caller&& caller, Args&&... args) -> std::tuple<decltype(caller(std::get<Seq>(t),std::forward<Args>(args)...))...> {
    /// return type is tuple of caller applied to each type in Types with args
    using return_type = std::tuple<decltype(caller(std::get<Seq>(t),std::forward<Args>(args)...))...>;
    return return_type(caller(std::get<Seq>(t),std::forward<Args>(args)...)...); // default construct result tuple
};

template <typename Caller, typename... Types, typename... Args>
auto call_all(std::tuple<Types...>& t, Caller&& caller, Args&&... args) -> std::tuple<decltype(caller(std::declval<Types>(),std::forward<Args>(args)...))...> {
    return call_all_helper(typename seq<sizeof...(Types)>::type(), t, std::forward<Caller>(caller), std::forward<Args>(args)...);
};

// const ref version (elements are const)
template <typename Caller, typename... Types, typename... Args>
auto call_all(const std::tuple<Types...>& t, Caller&& caller, Args&&... args) -> std::tuple<decltype(caller(std::declval<const Types>(),std::forward<Args>(args)...))...> {
    return call_all_helper(typename seq<sizeof...(Types)>::type(), t, std::forward<Caller>(caller), std::forward<Args>(args)...);
};

template <template<typename> class type_trait, typename Caller, typename... Types, typename... Args>
auto call_all_filtered(std::tuple<Types...>& t, Caller&& caller, Args&&... args)
   -> decltype(call_all_helper(typename indeces_of<type_trait, types<Types...>>::seq(), t, std::forward<Caller>(caller), std::forward<Args>(args)...)) {
    return call_all_helper(typename indeces_of<type_trait, types<Types...>>::seq(), t, std::forward<Caller>(caller), std::forward<Args>(args)...);
};

// this is a copy from above with `const` added
template <template<typename> class type_trait, typename Caller, typename... Types, typename... Args>
auto call_all_filtered(const std::tuple<Types...>& t, Caller&& caller, Args&&... args)
   -> decltype(call_all_helper(typename indeces_of<type_trait, types<Types...>>::seq(), t, std::forward<Caller>(caller), std::forward<Args>(args)...)) {
    return call_all_helper(typename indeces_of<type_trait, types<Types...>>::seq(), t, std::forward<Caller>(caller), std::forward<Args>(args)...);
};




// TODO: new flat_type implementation (compile time buffers, rather than the
//       shitty type-unsafe implementation we currently have)
// TODO: serialization (of std containers) implementation!


} // namespace mxx

#endif // MXX_EXPERIMENTAL_BUFFER
