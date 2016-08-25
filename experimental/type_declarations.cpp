
#include <cstddef>
#include <iostream>
#include <utility>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <limits>

#include <mxx/comm.hpp>
#include <mxx/datatypes.hpp>

#include "type_traits.hpp"
#include "member_tuple.hpp"

struct no_member {
    double d;
    int i;
};

struct static_constexpr_obj {
    int i;
    double d;
    //char x[4];

    static constexpr auto datatype = declare_type(&static_constexpr_obj::i, &static_constexpr_obj::d);
};


struct static_const_obj {
    double d;
    int i;
    typedef static_const_obj baz;
    const static std::tuple<int baz::*, double baz::*> datatype;
};

struct nonstatic_obj {
    int datatype;
};

struct nonstatic_func {
    void datatype() {
    }
};

struct static_func {
    static void datatype() {}
};

struct multiple_funcs {
    void datatype() { }
    void datatype(int) { }
    int datatype(int, int) { return 0;}
};

struct multiple_static_funcs {
    static void datatype() { }
    static void datatype(int) { }
    static int datatype(int, int) { return 0;}
};

struct multi_mixed_funcs {
    static void datatype() { }
    static void datatype(float) { }
    void datatype(int) { }
    int datatype(int, int) { return 0;}
};


// check if this has a constexpr member (TODO use laptop version)

// is_simple_datatype<T>: if it declares its datatype and if all members are simple
//                        builtin types are simple by default

// need checker for has static object member  (vs function)
// MXX_HAS_STATIC_OBJECT_MEMBER()

// buffer_tag, custom_buffer_tag, etc...

#define PRINT_TYPE_TRAIT(TRAIT, TYPE) std::cout << "" # TRAIT " for type " # TYPE ": " << TRAIT < TYPE > ::value << std::endl;

#define PRINT_TYPE_TRAIT_ALL(TRAIT) \
    PRINT_TYPE_TRAIT(TRAIT, no_member) \
    PRINT_TYPE_TRAIT(TRAIT, static_constexpr_obj) \
    PRINT_TYPE_TRAIT(TRAIT, static_const_obj) \
    PRINT_TYPE_TRAIT(TRAIT, nonstatic_obj) \
    PRINT_TYPE_TRAIT(TRAIT, static_func) \
    PRINT_TYPE_TRAIT(TRAIT, nonstatic_func) \
    PRINT_TYPE_TRAIT(TRAIT, multiple_funcs) \
    PRINT_TYPE_TRAIT(TRAIT, multiple_static_funcs) \
    PRINT_TYPE_TRAIT(TRAIT, multi_mixed_funcs)

int main(int argc, char *argv[])
{
    PRINT_TYPE_TRAIT_ALL(has_static_object_member)
    PRINT_TYPE_TRAIT_ALL(has_static_function_member)

    PRINT_TYPE_TRAIT_ALL(has_function_member)

    // test for the overloaded functions:
    std::cout << "has_function_member: void(): " << has_function_member<multiple_funcs, void()>::value << std::endl;
    std::cout << "has_function_member: void(): " << has_function_member<multi_mixed_funcs, void()>::value << std::endl;
    std::cout << "has_function_member: void(): " << has_function_member<multiple_static_funcs, void()>::value << std::endl;

    std::cout << "has_static_function_member: void(): " << has_static_function_member<multiple_funcs, void()>::value << std::endl;
    std::cout << "has_static_function_member: void(): " << has_static_function_member<multi_mixed_funcs, void()>::value << std::endl;
    std::cout << "has_static_function_member: void(): " << has_static_function_member<multiple_static_funcs, void()>::value << std::endl;

    std::cout << typeid(static_cast<void(multiple_funcs::*)()>(&multiple_funcs::datatype)).name() << std::endl;

    std::cout << typeid(*static_cast<void(*)()>(multiple_static_funcs::datatype)).name() << std::endl;

    //.PRINT_TYPE_TRAIT_ALL(has_datatype_constexpr)


    /*
    std::cout << typeid(decltype(foo::datatype)).name() << std::endl;
    std::cout << "is const: " << std::is_const<decltype(foo::datatype)>::value << std::endl;
    std::cout << "is const: " << std::is_const<decltype(baz::datatype)>::value << std::endl;


    std::cout << "all are member: " << all_are<std::is_member_pointer, std::remove_const<decltype(foo::datatype)>::type>::value << std::endl;
    std::cout << "all are member: " << all_are<std::is_member_pointer, std::tuple<int foo::*, double foo::*>>::value << std::endl;
    std::cout << "all are member: " << are_all_member_pointer<decltype(baz::datatype)>::value << std::endl;

    std::cout << "is member ptr: "<< std::is_member_pointer<std::tuple_element<0, decltype(foo::datatype)>::type>::value << std::endl;
    std::cout << "is member ptr: "<< std::is_member_pointer<std::tuple_element<1, decltype(foo::datatype)>::type>::value << std::endl;
    std::cout << "tuple size: " << std::tuple_size<decltype(foo::datatype)>::value << std::endl;

    std::cout << " are all arith: " << all_are<std::is_arithmetic, std::tuple<int, float, char>>::value << std::endl;
    */

    return 0;
}


