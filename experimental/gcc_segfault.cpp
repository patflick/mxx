#include <tuple>
#include <iostream>
#include <typeinfo>

template <typename... Types>
constexpr std::tuple<Types...> my_make_tuple(Types&&...t) {
    return std::tuple<Types...>(std::forward<Types>(t)...);
}

struct st {
    int i,j;
    double d;
    static constexpr auto x = my_make_tuple(&st::i,&st::j, &st::d);
};

int main() {
    st t;

    std::cout << typeid(st::x).name() << std::endl;

    //int st::* x = std::get<0>(st::x);
}
