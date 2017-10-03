/*
 * Copyright 2015 Georgia Institute of Technology
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

/**
 * @file    algos.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @author  Nagakishore Jammula <njammula3@mail.gatech.edu>
 * @author  Tony Pan <tpan7@gatech.edu>
 * @brief   Implements some common sequential algorithms
 *
 */

#ifndef MXX_ALGOS_H
#define MXX_ALGOS_H

#include <cstdlib>
#include <vector>

#include "common.hpp"

namespace mxx {

/**
 * @brief  Calculates the inclusive prefix sum of the given input range.
 *
 * @param begin An iterator to the beginning of the sequence.
 * @param end   An iterator to the end of the sequence.
 */
template <typename Iterator>
void prefix_sum(Iterator begin, const Iterator end)
{
    // set the total sum to zero
    typename std::iterator_traits<Iterator>::value_type sum = 0;

    // calculate the inclusive prefix sum
    while (begin != end)
    {
        sum += *begin;
        *begin = sum;
        ++begin;
    }
}

/**
 * @brief  Calculates the exclusive prefix sum of the given input range.
 *
 * @param begin An iterator to the beginning of the sequence.
 * @param end   An iterator to the end of the sequence.
 */
template <typename Iterator>
void excl_prefix_sum(Iterator begin, const Iterator end)
{
    // set the total sum to zero
    typename std::iterator_traits<Iterator>::value_type sum = 0;
    typename std::iterator_traits<Iterator>::value_type tmp;

    // calculate the exclusive prefix sum
    while (begin != end)
    {
        tmp = sum;
        sum += *begin;
        *begin = tmp;
        ++begin;
    }
}


/*********************************************************************
 *                       Bucketing algorithms                        *
 *********************************************************************/

// TODO: iterator version?

/**
 * @brief   Inplace bucketing of values into `num_buckets` buckets.
 *
 * This particular implementation uses an internal temporary buffer
 * of the same size as the input. Thus requiring that amount of additional
 * memory space. For a version that doesn't need O(n) additional memory,
 * use the (somewhat slower) `bucketing_inplace()` function below.
 *
 * @tparam T            Input type
 * @tparam Func         Type of the key function.
 * @param input[in|out] Contains the values to be bucketed.
 *                      This vector is both input and output.
 * @param key_func      A function taking a type T and returning the bucket index
 *                      in the range [0, num_buckets).
 * @param num_buckets   The total number of buckets.
 *
 * @return              The number of elements in each bucket.
 *                      The size of this vector is `num_buckets`.
 */
template <typename T, typename Func>
std::vector<size_t> bucketing(std::vector<T>& input, Func key_func, size_t num_buckets) {
    // initialize number of elements per bucket
    std::vector<size_t> bucket_counts(num_buckets, 0);

    // if input is empty, simply return
    if (input.size() == 0)
        return bucket_counts;

    // [1st pass]: counting the number of elements per bucket
    for (auto it = input.begin(); it != input.end(); ++it) {
        MXX_ASSERT(0 <= key_func(*it) && (size_t)key_func(*it) < num_buckets);
        bucket_counts[key_func(*it)]++;
    }

    // get offsets of where buckets start (= exclusive prefix sum)
    std::vector<std::size_t> offset(bucket_counts.begin(), bucket_counts.end());
    excl_prefix_sum(offset.begin(), offset.end());

    // [2nd pass]: saving elements into correct position
    std::vector<T> tmp_result(input.size());
    for (auto it = input.begin(); it != input.end(); ++it) {
        tmp_result[offset[key_func(*it)]++] = *it;
    }

    // replacing input with tmp result buffer and return the number of elements
    // in each bucket
    input.swap(tmp_result);
    return bucket_counts;
}

// inplace version (doesn't require O(n) additional memory like the other two approaches)
/**
 * @brief   Inplace bucketing of values into `num_buckets` buckets.
 *
 * This particular implementation is truly inplace, and doesn't require O(n)
 * additional memory like the `bucketing()` function above.
 * However, this implementation is slightly slower (~1.5x).
 *
 * @tparam T            Input type
 * @tparam Func         Type of the key function.
 * @param input[in|out] Contains the values to be bucketed.
 *                      This vector is both input and output.
 * @param key_func      A function taking a type T and returning the bucket index
 *                      in the range [0, num_buckets).
 * @param num_buckets   The total number of buckets.
 *
 * @return              The number of elements in each bucket.
 *                      The size of this vector is `num_buckets`.
 */
template <typename T, typename Func>
std::vector<size_t> bucketing_inplace(std::vector<T>& input, Func key_func, size_t num_buckets) {
    // initialize number of elements per bucket
    std::vector<size_t> bucket_counts(num_buckets, 0);

    // if input is empty, simply return
    if (input.size() == 0)
        return bucket_counts;

    // [1st pass]: counting the number of elements per bucket
    for (auto it = input.begin(); it != input.end(); ++it) {
        MXX_ASSERT(0 <= key_func(*it) && (size_t)key_func(*it) < num_buckets);
        bucket_counts[key_func(*it)]++;
    }
    // get exclusive prefix sum
    // get offsets of where buckets start (= exclusive prefix sum)
    // and end (=inclusive prefix sum)
    std::vector<size_t> offset(bucket_counts.begin(), bucket_counts.end());
    std::vector<size_t> upper_bound(bucket_counts.begin(), bucket_counts.end());
    excl_prefix_sum(offset.begin(), offset.end());
    prefix_sum(upper_bound.begin(), upper_bound.end());

    // in-place bucketing
    size_t cur_b = 0;
    for (size_t i = 0; i < input.size();) {
        // skip full buckets
        while (cur_b < num_buckets-1 && offset[cur_b] >= upper_bound[cur_b]) {
            // skip over full buckets
            i = offset[++cur_b];
        }
        // break if all buckets are done
        if (cur_b >= num_buckets-1)
            break;
        size_t target_b = key_func(input[i]);
        MXX_ASSERT(0 <= target_b && target_b < num_buckets);
        if (target_b == cur_b) {
            // item correctly placed
            ++i;
        } else {
            // swap to correct bucket
            MXX_ASSERT(target_b > cur_b);
            std::swap(input[i], input[offset[target_b]]);
        }
        offset[target_b]++;
    }
    return bucket_counts;
}

} // namespace mxx

#endif // MXX_ALGOS_H
