/*
 * Copyright 2016 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

// testing experimental type traits
#include <experimental/type_traits.hpp>


/*********************************************************************
 *    declaring a set of classes with different types of members     *
 *********************************************************************/

#define MEMBER_NAME test_member

// generate all member traits for member with name `test_member`
MXX_GENERATE_MEMBER_TRAITS(test_member)

struct no_member {
};

struct static_constexpr_obj {
    static constexpr auto MEMBER_NAME = 13*1.0;
};

struct static_const_obj {
    const static int MEMBER_NAME = 1;
};

struct static_obj {
    static int MEMBER_NAME;
};

struct nonstatic_obj {
    int MEMBER_NAME;
};

struct nonstatic_const_obj {
    const int MEMBER_NAME;
};

struct nonstatic_func {
    void MEMBER_NAME(int) const {}
};

struct static_func {
    static void MEMBER_NAME() {}
};

struct multiple_funcs {
    void MEMBER_NAME() { }
    void MEMBER_NAME(int) {}
    void MEMBER_NAME(int) const {}
    int MEMBER_NAME(int, int) { return 0;}
};

struct multiple_static_funcs {
    static void MEMBER_NAME() { }
    static void MEMBER_NAME(int) { }
    static int MEMBER_NAME(int, int) { return 0;}
};

struct multi_mixed_funcs {
    static void MEMBER_NAME() { }
    static void MEMBER_NAME(float) { }
    void MEMBER_NAME(int) { } // const to make sure it works also with cv qualified
    void MEMBER_NAME(int) const { } // const to make sure it works also with cv qualified
    int MEMBER_NAME(int, int) { return 0;}
};

struct typedef_member {
    typedef char MEMBER_NAME [2];
};

struct using_member {
    using MEMBER_NAME = float;
};


TEST(MxxExperimentalTypeTraits, static_objects) {
    // classes with static objects
    EXPECT_TRUE(has_static_member_object_test_member<static_obj>::value);
    EXPECT_TRUE(has_static_member_object_test_member<static_const_obj>::value);
    EXPECT_TRUE(has_static_member_object_test_member<static_constexpr_obj>::value);

    // classes without static objects
    EXPECT_FALSE(has_static_member_object_test_member<no_member>::value);
    EXPECT_FALSE(has_static_member_object_test_member<nonstatic_obj>::value);
    EXPECT_FALSE(has_static_member_object_test_member<nonstatic_func>::value);
    EXPECT_FALSE(has_static_member_object_test_member<static_func>::value);
    EXPECT_FALSE(has_static_member_object_test_member<multiple_funcs>::value);
    EXPECT_FALSE(has_static_member_object_test_member<multiple_static_funcs>::value);
    EXPECT_FALSE(has_static_member_object_test_member<multi_mixed_funcs>::value);
}

TEST(MxxExperimentalTypeTraits, nonstatic_objects) {
    // classes with non-static objects
    EXPECT_TRUE(has_member_object_test_member<nonstatic_obj>::value);

    // classes without non-static objects
    EXPECT_FALSE(has_member_object_test_member<static_obj>::value);
    EXPECT_FALSE(has_member_object_test_member<static_const_obj>::value);
    EXPECT_FALSE(has_member_object_test_member<static_constexpr_obj>::value);
    EXPECT_FALSE(has_member_object_test_member<no_member>::value);
    EXPECT_FALSE(has_member_object_test_member<nonstatic_func>::value);
    EXPECT_FALSE(has_member_object_test_member<static_func>::value);
    EXPECT_FALSE(has_member_object_test_member<multiple_funcs>::value);
    EXPECT_FALSE(has_member_object_test_member<multiple_static_funcs>::value);
    EXPECT_FALSE(has_member_object_test_member<multi_mixed_funcs>::value);
}

TEST(MxxExperimentalTypeTraits, unique_static_function) {
    // classes with unique static function
    EXPECT_TRUE(has_static_member_function_test_member<static_func>::value);

    // classes without unique static function
    EXPECT_FALSE(has_static_member_function_test_member<static_obj>::value);
    EXPECT_FALSE(has_static_member_function_test_member<static_const_obj>::value);
    EXPECT_FALSE(has_static_member_function_test_member<static_constexpr_obj>::value);
    EXPECT_FALSE(has_static_member_function_test_member<no_member>::value);
    EXPECT_FALSE(has_static_member_function_test_member<nonstatic_obj>::value);
    EXPECT_FALSE(has_static_member_function_test_member<nonstatic_func>::value);
    EXPECT_FALSE(has_static_member_function_test_member<multiple_funcs>::value);
    EXPECT_FALSE(has_static_member_function_test_member<multiple_static_funcs>::value);
    EXPECT_FALSE(has_static_member_function_test_member<multi_mixed_funcs>::value);
}

TEST(MxxExperimentalTypeTraits, typed_static_function) {
    // classes with static function void()
    EXPECT_TRUE((has_static_member_function_test_member<static_func, void()>::value));
    EXPECT_TRUE((has_static_member_function_test_member<multiple_static_funcs, void()>::value));
    EXPECT_TRUE((has_static_member_function_test_member<multi_mixed_funcs,void()>::value));

    // classes without static function void()
    EXPECT_FALSE((has_static_member_function_test_member<static_obj,void()>::value));
    EXPECT_FALSE((has_static_member_function_test_member<static_const_obj,void()>::value));
    EXPECT_FALSE((has_static_member_function_test_member<static_constexpr_obj,void()>::value));
    EXPECT_FALSE((has_static_member_function_test_member<no_member,void()>::value));
    EXPECT_FALSE((has_static_member_function_test_member<nonstatic_obj,void()>::value));
    EXPECT_FALSE((has_static_member_function_test_member<nonstatic_func,void()>::value));
    EXPECT_FALSE((has_static_member_function_test_member<multiple_funcs,void()>::value));
}

TEST(MxxExperimentalTypeTraits, unique_member_function) {
    // classes with unique member function
    EXPECT_TRUE(has_member_function_test_member<nonstatic_func>::value);

    // classes without unique member function
    EXPECT_FALSE(has_member_function_test_member<static_obj>::value);
    EXPECT_FALSE(has_member_function_test_member<static_const_obj>::value);
    EXPECT_FALSE(has_member_function_test_member<static_constexpr_obj>::value);
    EXPECT_FALSE(has_member_function_test_member<no_member>::value);
    EXPECT_FALSE(has_member_function_test_member<nonstatic_obj>::value);
    EXPECT_FALSE(has_member_function_test_member<static_func>::value);
    EXPECT_FALSE(has_member_function_test_member<multiple_funcs>::value);
    EXPECT_FALSE(has_member_function_test_member<multiple_static_funcs>::value);
    EXPECT_FALSE(has_member_function_test_member<multi_mixed_funcs>::value);
}

TEST(MxxExperimentalTypeTraits, typed_member_function) {
    // classes with member function void(int)
    EXPECT_TRUE((has_member_function_test_member<nonstatic_func,void(int) const>::value));
    EXPECT_TRUE((has_member_function_test_member<multiple_funcs,void(int)>::value));
    EXPECT_TRUE((has_member_function_test_member<const volatile multi_mixed_funcs,void(int)>::value));

    // classes without member function void(int)
    EXPECT_FALSE((has_member_function_test_member<static_obj,void(int)>::value));
    EXPECT_FALSE((has_member_function_test_member<static_const_obj,void(int)>::value));
    EXPECT_FALSE((has_member_function_test_member<static_constexpr_obj,void(int)>::value));
    EXPECT_FALSE((has_member_function_test_member<no_member,void(int)>::value));
    EXPECT_FALSE((has_member_function_test_member<nonstatic_obj,void(int)>::value));
    EXPECT_FALSE((has_member_function_test_member<static_func, void(int)>::value));
    EXPECT_FALSE((has_member_function_test_member<multiple_static_funcs, void(int)>::value));
}

TEST(MxxExperimentalTypeTraits, const_typed_member_function) {
    // classes with member function `void(int) const`
    EXPECT_TRUE((has_member_function_test_member<const nonstatic_func,void(int) const>::value));
    EXPECT_TRUE((has_member_function_test_member<multiple_funcs,void(int) const>::value));
    EXPECT_TRUE((has_member_function_test_member<multi_mixed_funcs,void(int) const>::value));

    // classes without member function `void(int) const`
    EXPECT_FALSE((has_member_function_test_member<static_obj,void(int) const>::value));
    EXPECT_FALSE((has_member_function_test_member<static_const_obj,void(int) const>::value));
    EXPECT_FALSE((has_member_function_test_member<static_constexpr_obj,void(int) const>::value));
    EXPECT_FALSE((has_member_function_test_member<no_member,void(int) const>::value));
    EXPECT_FALSE((has_member_function_test_member<static_func, void(int) const>::value));
    EXPECT_FALSE((has_member_function_test_member<multiple_static_funcs, void(int) const>::value));
    EXPECT_FALSE((has_member_function_test_member<nonstatic_obj,void(int) const>::value));
}

TEST(MxxExperimentalTypeTraits, typedef_member) {
    // classes with the typedef `test_member`
    EXPECT_TRUE(has_member_typedef_test_member<typedef_member>::value);
    EXPECT_TRUE(has_member_typedef_test_member<using_member>::value);

    // classes without typedef
    EXPECT_FALSE(has_member_typedef_test_member<nonstatic_func>::value);
    EXPECT_FALSE(has_member_typedef_test_member<static_obj>::value);
    EXPECT_FALSE(has_member_typedef_test_member<static_const_obj>::value);
    EXPECT_FALSE(has_member_typedef_test_member<static_constexpr_obj>::value);
    EXPECT_FALSE(has_member_typedef_test_member<no_member>::value);
    EXPECT_FALSE(has_member_typedef_test_member<nonstatic_obj>::value);
    EXPECT_FALSE(has_member_typedef_test_member<static_func>::value);
    EXPECT_FALSE(has_member_typedef_test_member<multiple_funcs>::value);
    EXPECT_FALSE(has_member_typedef_test_member<multiple_static_funcs>::value);
    EXPECT_FALSE(has_member_typedef_test_member<multi_mixed_funcs>::value);
}

