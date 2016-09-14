#ifndef MXX_EXPERIMENTAL_TYPE_TRAITS
#define MXX_EXPERIMENTAL_TYPE_TRAITS

#include <type_traits>
#include <tuple> // for all_are tuple specializations

#define MXX_GENERATE_HAS_STATIC_MEMBER_OBJECT(MEMBER_NAME)                     \
template <typename T, typename Enable = void>                                  \
struct has_static_member_object_ ## MEMBER_NAME : std::false_type {};          \
                                                                               \
template <typename T>                                                          \
struct has_static_member_object_ ## MEMBER_NAME <T, typename std::enable_if<   \
    !std::is_member_pointer<decltype(&T:: MEMBER_NAME )>::value                \
    && !std::is_function<decltype(T :: MEMBER_NAME )>::value>::type>           \
    : std::true_type {};                                                       \


#define MXX_GENERATE_HAS_MEMBER_OBJECT(MEMBER_NAME)                            \
template <typename T, typename Enable = void>                                  \
struct has_member_object_ ## MEMBER_NAME : std::false_type {};                 \
                                                                               \
template <typename T>                                                          \
struct has_member_object_ ## MEMBER_NAME <T, typename std::enable_if<          \
    std::is_member_object_pointer<decltype(&T:: MEMBER_NAME )>::value>::type>  \
    : std::true_type {};                                                       \
                                                                               \


#define MXX_GENERATE_HAS_STATIC_MEMBER_FUNCTION(MEMBER_NAME)                   \
template <typename T, typename Sig = std::false_type, typename Enable = void>  \
struct has_static_member_function_ ## MEMBER_NAME : std::false_type {};        \
                                                                               \
template <typename T>                                                          \
struct has_static_member_function_ ## MEMBER_NAME                              \
    <T, std::false_type, typename std::enable_if<                              \
    std::is_function<decltype(T:: MEMBER_NAME )>::value>::type>                \
    : std::true_type {};                                                       \
                                                                               \
template <typename T, typename R, typename... Args>                            \
struct has_static_member_function_ ## MEMBER_NAME                              \
    <T, R(Args...), typename std::enable_if<                                   \
    std::is_same<decltype(static_cast<R(*)(Args...)>(&T:: MEMBER_NAME )),      \
                          R(*)(Args...)>::value>::type>                        \
    : std::true_type {};                                                       \


#define MXX_GENERATE_HAS_MEMBER_FUNCTION(MEMBER_NAME)                          \
template <typename T, typename Sig = std::false_type, typename Enable = void>  \
struct has_member_function_ ## MEMBER_NAME : std::false_type {};               \
                                                                               \
template <typename T>                                                          \
struct has_member_function_ ## MEMBER_NAME                                     \
    <T, std::false_type, typename std::enable_if<                              \
    std::is_member_function_pointer<decltype(&T:: MEMBER_NAME )>::value>::type>\
    : std::true_type {};                                                       \
                                                                               \
template <typename T, typename Func>                                           \
struct has_member_function_ ## MEMBER_NAME <T, Func, typename std::enable_if<  \
    std::is_member_function_pointer<                                           \
        decltype(static_cast<Func T::*>(&T:: MEMBER_NAME ))>::value>::type>    \
    : std::true_type {};                                                       \
                                                                               \


#define MXX_GENERATE_HAS_MEMBER_TYPEDEF(TYPE_NAME)                             \
template <typename T, typename Enable = void>                                  \
struct has_member_typedef_ ## TYPE_NAME : std::false_type {};                  \
                                                                               \
template <typename T>                                                          \
struct has_member_typedef_ ## TYPE_NAME <T, typename std::enable_if<           \
    !std::is_same<typename T:: TYPE_NAME ,void>::value>::type>                 \
    : std::true_type {};                                                       \


#define MXX_GENERATE_MEMBER_TRAITS(MEMBER_NAME)                                \
        MXX_GENERATE_HAS_MEMBER_OBJECT(MEMBER_NAME)                            \
        MXX_GENERATE_HAS_STATIC_MEMBER_OBJECT(MEMBER_NAME)                     \
        MXX_GENERATE_HAS_MEMBER_FUNCTION(MEMBER_NAME)                          \
        MXX_GENERATE_HAS_STATIC_MEMBER_FUNCTION(MEMBER_NAME)                   \
        MXX_GENERATE_HAS_MEMBER_TYPEDEF(MEMBER_NAME)                           \


/*
template <typename T, typename Enable = void>
struct has_datatype_constexpr : std::false_type {};

template <typename T>
struct has_datatype_constexpr<T,typename std::enable_if<
     std::is_member_object_pointer<typename std::tuple_element<0,decltype(T::datatype)>::type>::value
// the standard § 4.11 [conv.mem]:
//    guarantuees that `valid` pointer-to-member are always distinct from the
//    `null pointer to member`
&& std::get<0>(T::datatype) != nullptr
>::type> : std::true_type {};
*/

template <template <typename> class Trait, typename... Types>
struct all_are;

template <template<typename> class Trait, typename Type, typename... Types>
struct all_are<Trait, Type, Types...> : std::integral_constant<bool,
    Trait<Type>::value && all_are<Trait, Types...>::value> {};

template <template<typename> class Trait, typename Type>
struct all_are<Trait, Type> : std::integral_constant<bool,
    Trait<Type>::value> {};

template <template <typename> class Trait, typename... Types>
struct any_are;

template <template<typename> class Trait, typename Type, typename... Types>
struct any_are<Trait, Type, Types...> : std::integral_constant<bool,
    Trait<Type>::value || any_are<Trait, Types...>::value> {};

template <template<typename> class Trait, typename Type>
struct any_are<Trait, Type> : std::integral_constant<bool,
    Trait<Type>::value> {};

// specialization for std::tuple
template <template<typename> class Trait, typename... Types>
struct all_are<Trait, std::tuple<Types...>> : std::integral_constant<bool,
    all_are<Trait, Types...>::value> {};

template <typename... T>
struct are_all_member_pointer : std::false_type {};

template <typename... Types>
struct are_all_member_pointer<std::tuple<Types...>> : all_are<std::is_member_pointer, Types...> {};

template <typename... Types>
struct are_all_member_pointer<const std::tuple<Types...>> : all_are<std::is_member_pointer, Types...> {};

#endif // MXX_EXPERIMENTAL_TYPE_TRAITS
