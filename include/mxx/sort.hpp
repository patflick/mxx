/**
 * @file    sort.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements parallel sorting.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef MXX_SORT_HPP
#define MXX_SORT_HPP

#include "samplesort.hpp"

namespace mxx {

template<typename _Iterator, typename _Compare>
void sort(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD, bool _AssumeBlockDecomp = true)
{
    // use sample sort
    impl::samplesort<_Iterator, _Compare, false>(begin, end, comp, comm, _AssumeBlockDecomp);
}

template<typename _Iterator, typename _Compare>
void stable_sort(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD, bool _AssumeBlockDecomp = true)
{
    // use stable sample sort
    impl::samplesort<_Iterator, _Compare, true>(begin, end, comp, comm, _AssumeBlockDecomp);
}

template<typename _Iterator, typename _Compare>
bool is_sorted(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD)
{
    return impl::is_sorted(begin, end, comp, comm);
}

} // namespace mxx

#endif // MXX_SORT_HPP
