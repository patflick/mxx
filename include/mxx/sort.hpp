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
 * @file    sort.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the interface for parallel sorting.
 *
 * TODO:
 * - [ ] fix stable sort
 * - [ ] fix sort on non GCC compilers
 * - [ ] radix sort
 * - [ ] fix sorting of samples
 * - [ ] implement and try out different parallel sorting algorithms
 * - [ ] bitonic sort for single elemenet per processor
 */

#ifndef MXX_SORT_HPP
#define MXX_SORT_HPP

#include "comm_fwd.hpp"
#include "samplesort.hpp"

namespace mxx {

template<typename _Iterator, typename _Compare>
void sort(_Iterator begin, _Iterator end, _Compare comp, const mxx::comm& comm = mxx::comm(), bool _AssumeBlockDecomp = true) {
    // use sample sort
    impl::samplesort<_Iterator, _Compare, false>(begin, end, comp, comm, _AssumeBlockDecomp);
}

template<typename _Iterator, typename _Compare>
void stable_sort(_Iterator begin, _Iterator end, _Compare comp, const mxx::comm& comm = mxx::comm(), bool _AssumeBlockDecomp = true) {
    // use stable sample sort
    impl::samplesort<_Iterator, _Compare, true>(begin, end, comp, comm, _AssumeBlockDecomp);
}

template<typename _Iterator, typename _Compare>
bool is_sorted(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD)
{
    return impl::is_sorted(begin, end, comp, comm);
}

// assumes input is sorted, removes duplicates in global range
template <typename ForwardIt, typename BinaryPredicate>
ForwardIt unique(ForwardIt begin, ForwardIt end, BinaryPredicate eq, const mxx::comm& comm) {
    typedef typename std::iterator_traits<ForwardIt>::value_type T;
    ForwardIt dest = begin;
    mxx::comm c = comm.split(begin != end);
    comm.with_subset(begin != end, [&](const mxx::comm& c) {
        size_t n = std::distance(begin, end);

        // send last item to next processor
        T last = *(begin + (n-1));
        T prev = mxx::right_shift(last, c);

        // skip elements which are equal to the last one on the previous processor
        while (eq(prev, *begin))
            ++begin;
        *dest = *begin;

        // remove duplicates
        while (++begin != end)
            if (!eq(*dest, *begin))
                *++dest = *begin;
        ++dest;
    });
    return dest;
}

#include "comm_def.hpp"

} // namespace mxx

#endif // MXX_SORT_HPP
