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
 * @file    samplesort.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements parallel, MPI sample sort
 */

#ifndef MXX_SAMPLESORT_HPP
#define MXX_SAMPLESORT_HPP

#include <mpi.h>

#include <assert.h>

#include <iterator>
#include <algorithm>
#include <vector>
#include <limits>

// for multiway-merge
// TODO: impelement own in case it is not GNU C++
#include <parallel/multiway_merge.h>
#include <parallel/merge.h>

#include "partition.hpp"
#include "datatypes.hpp"
#include "collective.hpp"
#include "shift.hpp"
#include "distribution.hpp"
#include "timer.hpp"


#define SS_ENABLE_TIMER 0
#if SS_ENABLE_TIMER
#define SS_TIMER_START(comm) mxx::section_timer timer(std::cerr, comm, 0);
#define SS_TIMER_END_SECTION(str) timer.end_section(str);
#else
#define SS_TIMER_START(comm)
#define SS_TIMER_END_SECTION(str)
#endif

#define MEASURE_LOAD_BALANCE 0

namespace mxx {
namespace impl {

template<typename _Iterator, typename _Compare>
bool is_sorted(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get value type of underlying data
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    if (p == 1)
        return std::is_sorted(begin, end, comp);

    // check that it is locally sorted (int for MPI_Reduction)
    int sorted = std::is_sorted(begin, end, comp);

    // compare if last element on left processor is not bigger than first
    // element on mine
    value_type left_el = mxx::right_shift(*(end-1));

    // check if sorted
    if (rank > 0) {
        sorted = sorted && !comp(*begin, left_el);
    }

    // get global minimum to determine if the whole sequence is sorted
    int all_sorted;
    MPI_Allreduce(&sorted, &all_sorted, 1, MPI_INT, MPI_MIN, comm);

    // return as boolean
    return (all_sorted > 0);
}

template <typename _Iterator, typename _Compare>
std::vector<typename std::iterator_traits<_Iterator>::value_type>
sample_arbit_decomp(_Iterator begin, _Iterator end, _Compare comp, int s, MPI_Comm comm, MPI_Datatype mpi_dt)
{
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;
    std::size_t local_size = std::distance(begin, end);

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // get total size n
    std::size_t total_size;
    // get MPI type
    mxx::datatype<std::size_t> size_dt;
    MPI_Datatype mpi_size_t = size_dt.type();

    MPI_Allreduce(&local_size, &total_size, 1, mpi_size_t, MPI_SUM, comm);

    //  pick a total of s*p samples, thus locally pick ceil((local_size/n)*s*p)
    //  and at least one samples from each processor.
    //  this will result in at least s*p samples.
    std::size_t local_s;
    if (local_size == 0)
        local_s = 0;
    else
        local_s = std::max<std::size_t>(((local_size*s*p)+total_size-1)/total_size, 1);

    //. init samples
    std::vector<value_type> local_splitters;

    // pick local samples
    if (local_s > 0)
    {
        local_splitters.resize(local_s);
        _Iterator pos = begin;
        for (std::size_t i = 0; i < local_splitters.size(); ++i)
        {
            std::size_t bucket_size = local_size / (local_s+1) + (i < (local_size % (local_s+1)) ? 1 : 0);
            // pick last element of each bucket
            pos += (bucket_size-1);
            local_splitters[i] = *pos;
            ++pos;
        }
    }

    // 2. gather samples to `rank = 0`
    // - TODO: rather call sample sort
    //         recursively and implement a base case for samplesort which does
    //         gather to rank=0, local sort and redistribute
    std::vector<value_type> all_samples = gather_vectors(local_splitters, comm);

    // sort and pick p-1 samples on master
    if (rank == 0)
    {
        // 3. local sort on master
        std::sort(all_samples.begin(), all_samples.end(), comp);

        // 4. pick p-1 splitters and broadcast them
        if (local_splitters.size() != p-1)
        {
            local_splitters.resize(p-1);
        }
        // split into `p` pieces and choose the `p-1` splitting elements
        _Iterator pos = all_samples.begin();
        for (std::size_t i = 0; i < local_splitters.size(); ++i)
        {
            std::size_t bucket_size = (p*s) / p + (i < static_cast<std::size_t>((p*s) % p) ? 1 : 0);
            // pick last element of each bucket
            local_splitters[i] = *(pos + (bucket_size-1));
            pos += bucket_size;
        }
    }

    // size splitters for receiving
    if (local_splitters.size() != p-1)
    {
        local_splitters.resize(p-1);
    }

    // 4. broadcast and receive final splitters
    MPI_Bcast(&local_splitters[0], local_splitters.size(), mpi_dt, 0, comm);

    return local_splitters;
}


template <typename _Iterator, typename _Compare>
std::vector<typename std::iterator_traits<_Iterator>::value_type>
sample_block_decomp(_Iterator begin, _Iterator end, _Compare comp, int s, MPI_Comm comm, MPI_Datatype mpi_dt)
{
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;
    std::size_t local_size = std::distance(begin, end);
    assert(local_size > 0);

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);


    // 1. samples
    //  - pick `s` samples equally spaced such that `s` samples define `s+1`
    //    subsequences in the sorted order
    std::vector<value_type> local_splitters(s);
    _Iterator pos = begin;
    for (std::size_t i = 0; i < local_splitters.size(); ++i)
    {
        std::size_t bucket_size = local_size / (s+1) + (i < (local_size % (s+1)) ? 1 : 0);
        // pick last element of each bucket
        pos += (bucket_size-1);
        local_splitters[i] = *pos;
        ++pos;
    }

    // 2. gather samples to `rank = 0`
    // - TODO: rather call sample sort
    //         recursively and implement a base case for samplesort which does
    //         gather to rank=0, local sort and redistribute
    if (rank == 0)
    {
        std::vector<value_type> all_samples(p*s);
        MPI_Gather(&local_splitters[0], s, mpi_dt,
                   &all_samples[0], s, mpi_dt, 0, comm);

        // 3. local sort on master
        std::sort(all_samples.begin(), all_samples.end(), comp);

        // 4. pick p-1 splitters and broadcast them
        if (local_splitters.size() != p-1)
        {
            local_splitters.resize(p-1);
        }
        // split into `p` pieces and choose the `p-1` splitting elements
        _Iterator pos = all_samples.begin();
        for (std::size_t i = 0; i < local_splitters.size(); ++i)
        {
            std::size_t bucket_size = (p*s) / p + (i < static_cast<std::size_t>((p*s) % p) ? 1 : 0);
            // pick last element of each bucket
            local_splitters[i] = *(pos + (bucket_size-1));
            pos += bucket_size;
        }
    }
    else
    {
        // simply send
        MPI_Gather(&local_splitters[0], s, mpi_dt, NULL, 0, mpi_dt, 0, comm);

        // resize splitters for receiving
        if (local_splitters.size() != p-1)
        {
            local_splitters.resize(p-1);
        }
    }

    // 4. broadcast and receive final splitters
    MPI_Bcast(&local_splitters[0], local_splitters.size(), mpi_dt, 0, comm);

    return local_splitters;
}


template<typename _Iterator, typename _Compare, bool _Stable = false>
void samplesort(_Iterator begin, _Iterator end, _Compare comp, MPI_Datatype mpi_dt, MPI_Comm comm = MPI_COMM_WORLD, bool _AssumeBlockDecomp = true)
{
    // get value type of underlying data
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;


    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    SS_TIMER_START(comm);

    // perform local (stable) sorting
    if (_Stable)
        std::stable_sort(begin, end, comp);
    else
        std::sort(begin, end, comp);

    if (p == 1)
        return;

#if SS_ENABLE_TIMER
    MPI_Barrier(comm);
#endif
    SS_TIMER_END_SECTION("local_sort");


    // number of samples
    int s = p-1;
    // local size
    std::size_t local_size = std::distance(begin, end);

#ifndef NDEBUG
    // Assert the data is actually block decomposed when given that it is
    std::size_t global_size = mxx::allreduce(local_size, comm);
    partition::block_decomposition<std::size_t> mypart(global_size, p, rank);
    assert(!_AssumeBlockDecomp || local_size == mypart.local_size());
#endif
    // sample sort
    // 1. pick `s` samples on each processor
    // 2. gather to `rank=0`
    // 3. local sort on master
    // 4. broadcast the p-1 final splitters
    // 5. locally find splitter positions in data
    //    (if an identical splitter appears twice, then split evenly)
    //    => send_counts
    // 6. distribute send_counts with all2all to get recv_counts
    // 7. allocate enough space (may be more than previously allocated) for receiving
    // 8. all2all
    // 9. local reordering
    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by elements to neighbors

    // get splitters, using the method depending on whether the input consists
    // of arbitrary decompositions or not
    std::vector<value_type> local_splitters;
    if(_AssumeBlockDecomp)
        local_splitters = sample_block_decomp(begin, end, comp, s, comm, mpi_dt);
    else
        local_splitters = sample_arbit_decomp(begin, end, comp, s, comm, mpi_dt);
    SS_TIMER_END_SECTION("get_splitters");

    // 5. locally find splitter positions in data
    //    (if an identical splitter appears at least three times (or more),
    //    then split the intermediary buckets evenly) => send_counts
    std::vector<int> send_counts(p, 0);
    _Iterator pos = begin;
    partition::block_decomposition<std::size_t> local_part(local_size, p, rank);
    for (std::size_t i = 0; i < local_splitters.size();)
    {
        // get the number of splitters which are equal starting from `i`
        unsigned int split_by = 1;
        while (i+split_by < local_splitters.size()
               && !comp(local_splitters[i], local_splitters[i+split_by]))
        {
            ++split_by;
        }

        // get the range of equal elements
        std::pair<_Iterator, _Iterator> eqr = std::equal_range(pos, end, local_splitters[i], comp);

        // assign smaller elements to processor left of splitter (= `i`)
        send_counts[i] += std::distance(pos, eqr.first);
        pos = eqr.first;

        // split equal elements fairly across processors
        std::size_t eq_size = std::distance(pos, eqr.second);
        // try to split approx equal:
        std::size_t eq_size_split = (eq_size + send_counts[i]) / (split_by+1) + 1;
        for (unsigned int j = 0; j < split_by; ++j)
        {
            // TODO: this kind of splitting is not `stable` (need other strategy
            // to mak such splitting stable across processors)
            std::size_t out_size = 0;
            if ((std::size_t)send_counts[i+j] < local_part.local_size(i+j))
            {
                // try to distribute fairly
                out_size = std::min(std::max(local_part.local_size(i+j) - send_counts[i+j], eq_size_split), eq_size);
                eq_size -= out_size;
            }
            assert(out_size + send_counts[i+j] < (std::size_t)std::numeric_limits<int>::max());
            send_counts[i+j] += static_cast<int>(out_size);
        }
        // assign remaining elements to next processor
        assert(eq_size + send_counts[i+split_by] < (std::size_t)std::numeric_limits<int>::max());
        send_counts[i+split_by] += static_cast<int>(eq_size);
        i += split_by;
        pos = eqr.second;
    }
    // send last elements to last processor
    std::size_t out_size = std::distance(pos, end);
    assert(out_size < std::numeric_limits<int>::max());
    send_counts[p-1] += static_cast<int>(out_size);
    assert(std::accumulate(send_counts.begin(), send_counts.end(), 0) == local_size);

    SS_TIMER_END_SECTION("send_counts");

    /*
    // 6. distribute send_counts with all2all to get recv_counts
    std::vector<int> recv_counts = all2allv_get_recv_counts(send_counts, comm);

    // 7. allocate enough space (may be more than previously allocated) for receiving
    std::size_t recv_n = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
    assert(!_AssumeBlockDecomp || recv_n <= 2* local_size);
    std::vector<value_type> recv_elements(recv_n);

    // 8. all2all
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<int> recv_displs = get_displacements(recv_counts);
    SS_TIMER_END_SECTION("all2all_params");
    MPI_Alltoallv(&(*begin), &send_counts[0], &send_displs[0], mpi_dt,
                  &recv_elements[0], &recv_counts[0], &recv_displs[0], mpi_dt,
                  comm);
    */
    std::vector<int> recv_counts = all2all(send_counts, 1, comm);
    std::vector<int> recv_displs = get_displacements(recv_counts);
    std::size_t recv_n = recv_displs[p-1] + recv_counts[p-1];
    assert(!_AssumeBlockDecomp || (local_size <= 2 || recv_n <= 2* local_size));
    std::vector<value_type> recv_elements(recv_n);
    all2all(begin, recv_elements.begin(), send_counts, recv_counts, comm);
    SS_TIMER_END_SECTION("all2all");

    // 9. local reordering
    /*
    if (_Stable)
        std::stable_sort(recv_elements.begin(), recv_elements.end(), comp);
    else
        std::sort(recv_elements.begin(), recv_elements.end(), comp);
    */
    /* multiway-merge (using the implementation in __gnu_parallel) */
    // prepare the sequence offsets
    typedef typename std::vector<value_type>::iterator val_it;
    std::vector<std::pair<val_it, val_it> > seqs(p);
    for (int i = 0; i < p; ++i)
    {
        seqs[i].first = recv_elements.begin() + recv_displs[i];
        seqs[i].second = seqs[i].first + recv_counts[i];
    }
    val_it start_merge_it = recv_elements.begin();

    std::size_t merge_n = local_size;
    value_type* merge_buf_begin = &(*begin);
    std::vector<value_type> merge_buf;
    // TODO: reasonable values for the buffer?
    // currently: at least 1/10 th of the size to merge or 1 MiB
    if (local_size == 0 || local_size < recv_n / 10)
    {
        // at least 1MB buffer
        merge_n = std::max<std::size_t>(recv_n / 10, (1024*1024)/sizeof(value_type));
        merge_buf.resize(merge_n);
        merge_buf_begin = &merge_buf[0];
    }
    for (; recv_n > 0;)
    {
        if (recv_n < merge_n)
            merge_n = recv_n;
        // i)   merge at most `local_size` many elements sequentially
        __gnu_parallel::sequential_tag seq_tag;
        __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), merge_buf_begin, merge_n, comp, seq_tag);

        // ii)  compact the remaining elements in `recv_elements`
        for (int i = p-1; i > 0; --i)
        {
            seqs[i-1].first = std::copy_backward(seqs[i-1].first, seqs[i-1].second, seqs[i].first);
            seqs[i-1].second = seqs[i].first;
        }
        // iii) copy the output buffer `local_size` elements back into
        //      `recv_elements`
        start_merge_it = std::copy(merge_buf_begin, merge_buf_begin + merge_n, start_merge_it);
        assert(start_merge_it == seqs[0].first);

        // reduce the number of elements to be merged
        recv_n -= merge_n;
    }
    // clean up
    merge_buf.clear(); merge_buf.shrink_to_fit();

#if SS_ENABLE_TIMER
    MPI_Barrier(comm);
#endif
    SS_TIMER_END_SECTION("local_merge");

    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by elements to neighbors
    //    and save elements into the original iterator positions
    if (_AssumeBlockDecomp)
        redo_block_decomposition(recv_elements.begin(), recv_elements.end(), begin, comm);
    else
        redo_arbit_decomposition(recv_elements.begin(), recv_elements.end(), begin, local_size, comm);

    SS_TIMER_END_SECTION("fix_partition");
}

template<typename _Iterator, typename _Compare, bool _Stable = false>
void samplesort(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD, bool _AssumeBlockDecomp = true)
{
    // get value type of underlying data
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;

    // get MPI type
    mxx::datatype<value_type> dt;
    MPI_Datatype mpi_dt = dt.type();

    // sort
    samplesort(begin, end, comp, mpi_dt, comm, _AssumeBlockDecomp);
}

} // namespace impl
} // namespace mxx

#endif // MXX_SAMPLESORT_HPP


