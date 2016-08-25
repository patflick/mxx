/*
 * Custom tuple implementation
 */

#ifndef MXX_MEMBER_TUPLE
#define MXX_MEMBER_TUPLE

#include <cstddef> // for size_t
#include <utility> // for std::forward

template <typename T>
struct type_wrap {
    typedef T type;
};

template <size_t N, typename U>
struct member {
    U m;

    constexpr member() : m() {};
    constexpr member(U&& u) : m(u) {};
    constexpr member(const U& u) : m(u) {};

    U& get() {
        return m;
    }
    const U& get() const {
        return m;
    }
};

template <size_t I, typename... Types>
struct type_at;

template <typename Type>
struct type_at<0, Type> : type_wrap<Type> {};

template <typename Type, typename... Types>
struct type_at<0, Type, Types...> : type_wrap<Type> {};

template <size_t I, typename Type, typename... Types>
struct type_at<I, Type, Types...> : type_at<I-1, Types...> {};


// template sequence of size_t
template <size_t... S>
struct seq_ {};

template <size_t N, size_t I = 0, typename Seq = seq_<>>
struct seq;

template <size_t N, size_t I, size_t... Seq>
struct seq<N, I, seq_<Seq...>> : seq<N, I+1, seq_<Seq..., I>> {};

template <size_t N, size_t... Seq>
struct seq<N, N, seq_<Seq...>> : type_wrap<seq_<Seq...>> {};

template <typename Seq, typename... Types>
struct member_tuple_impl {};

// using flat multiple inheritance
// and unpacking of the sequence
template <size_t... Seq, typename... Types>
struct member_tuple_impl<seq_<Seq...>, Types...> : member<Seq, Types>... {
    constexpr member_tuple_impl() : member<Seq, Types>()... {}
    constexpr member_tuple_impl(Types&&...t) : member<Seq, Types>(std::forward<Types>(t))... {}
};

template <typename... Types>
struct member_tuple : member_tuple_impl<typename seq<sizeof...(Types)>::type, Types...> {

    using base_type = member_tuple_impl<typename seq<sizeof...(Types)>::type, Types...>;

    template <size_t I>
    typename type_at<I, Types...>::type& get() {
        return member<I, typename type_at<I, Types...>::type>::get();
    }
    template <size_t I>
    const typename type_at<I, Types...>::type& get() const {
        return member<I, typename type_at<I, Types...>::type>::get();
    }

    constexpr member_tuple() : base_type() {};
    constexpr member_tuple(Types&&... t) : base_type(std::forward<Types>(t)...) {};
};

/*
template <typename... Types>
constexpr member_tuple<size_t, Types...> declare_type(Types&&... t) {
    return member_tuple<size_t, Types...>(sizeof...(Types),std::forward<Types>(t)...);
};
*/

template <typename... Types>
constexpr std::tuple<Types...> declare_type(Types&&... t) {
    return std::tuple<Types...>(std::forward<Types>(t)...);
};

#endif // MXX_MEMBER_TUPLE
