
#include <iostream>

#include <mxx/env.hpp>
#include <mxx/datatypes.hpp>
#include "new_datatype.hpp"
#include "type_traits.hpp"

#include <cxx-prettyprint/prettyprint.hpp>

#include <cxxabi.h>


struct X {
    char c;
    double d;
    int i;

    static constexpr auto datatype = std::make_tuple(&X::i, &X::c, &X::d);
};

struct Y {
    int i;
};


template <typename T>
typename std::enable_if<mxx::is_simple_type<T>::value, void>::type test_datatype() {
    mxx::datatype dt = mxx::make_datatype<T>();
    std::cout << "================================================" << std::endl;
    std::cout << "Testing type: " << typeid(T).name() << std::endl;
    std::cout << "Sizeof(): " << sizeof(T) << std::endl;
    std::cout << "Extent: " << dt.get_extent() << std::endl;
    std::cout << "True extent: " << dt.get_true_extent() << std::endl;
    std::cout << "is simple: " << mxx::is_simple_type<T>::value << std::endl;

    // unpack test
    mxx::flat_repr r;
    r.type = dt.type();
    mxx::type_decoder::unpack_envelope(dt.type(), r);

    std::cout << "Pictogram:" << std::endl;
    r.print_pictogram();
}

template <typename T>
typename std::enable_if<!mxx::has_type_descriptor<T>::value, void>::type
test_descriptor() {
}

template <typename T>
typename std::enable_if<mxx::has_type_descriptor<T>::value, void>::type
test_descriptor() {
    std::cout << "all are builtin: " << all_are<mxx::is_builtin_type, decltype(mxx::get_datatype_descriptor<T>())>::value << std::endl;
    std::cout << "are all simple: " << all_are<mxx::is_simple_member, decltype(mxx::get_datatype_descriptor<T>())>::value << std::endl;
}

template <typename T>
typename std::enable_if<!mxx::is_simple_type<T>::value, void>::type test_datatype() {
    std::cout << "================================================" << std::endl;
    std::cout << "Testing type: " << typeid(T).name() << std::endl;
    std::cout << "Sizeof(): " << sizeof(T) << std::endl;
    std::cout << "is simple: " << mxx::is_simple_type<T>::value << std::endl;
    std::cout << "not simple because: " << std::endl;
    std::cout << "has descriptor: " << mxx::has_type_descriptor<T>::value << std::endl;
    test_descriptor<T>();
}


std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}

int main(int argc, char *argv[]) {
    mxx::env e(argc, argv);

    /*
    test_datatype<int>();
    test_datatype<float>();
    */

    test_datatype<std::pair<int, char>>();

    test_datatype<std::tuple<double, char>>();

    using rec_tuple = std::tuple<std::pair<int, char>, std::tuple<double, float>>;
    test_datatype<rec_tuple>();

    test_datatype<std::tuple<double, float>>();

    test_datatype<X>();

    test_datatype<Y>();
    test_datatype<std::tuple<int, Y>>();
    test_datatype<std::tuple<std::pair<int, int>, std::string>>();

    std::cout << "all builtin: " << all_are<mxx::is_builtin_type, std::tuple<float, int, float, double>>::value << std::endl;

    std::cout << "is simple type: " << mxx::is_simple_type_helper<X>::value << std::endl;
    std::cout << "is simple type: " << mxx::is_simple_type<float>::value << std::endl;
    std::cout << "is simple member: " << mxx::is_simple_member<float*>::value << std::endl;
    std::cout << "is simple member: " << mxx::is_simple_member<double X::*>::value << std::endl;
    typedef decltype(X::datatype) my_tuple;
    // TODO: we have a cv issue!
    std::cout << "all simple member: " << all_are<mxx::is_simple_member, std::remove_cv<my_tuple>::type>::value << std::endl;
    std::cout << "all simple member: " << all_are<mxx::is_simple_member, std::tuple<int X::*, char X::*, double X::*>>::value << std::endl;

    std::cout << "datatype type: " << demangle(typeid(const my_tuple).name()) << std::endl;
    return 0;
}
