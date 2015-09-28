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
 * @file    distribution.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements functions to redistribute data across processes.
 */

#ifndef MXX_DISTRIBUTION_HPP
#define MXX_DISTRIBUTION_HPP

#include <mpi.h>

#include <vector>
#include <queue>
#include "assert.h"

#include "datatypes.hpp"
#include "partition.hpp"
#include "reduction.hpp"
#include "collective.hpp"


#define MEASURE_LOAD_BALANCE 0

namespace mxx
{

/**
 * @brief Fixes an unequal distribution into a block decomposition
 */
template<typename _InIterator, typename _OutIterator>
void redo_block_decomposition(_InIterator begin, _InIterator end, _OutIterator out, const mxx::comm& comm)
{
    // if there's only one process, return a copy
    if (comm.size() == 1) {
        std::copy(begin, end, out);
        return;
    }

    // get local size
    std::size_t local_size = std::distance(begin, end);
    // get prefix sum of size and total size
    std::size_t prefix = mxx::exscan(local_size, comm);
    std::size_t total_size = mxx::allreduce(local_size, comm);

    // calculate where to send elements
    std::vector<size_t> send_counts(comm.size(), 0);
    partition::block_decomposition<std::size_t> part(total_size, comm.size(), comm.rank());
    int first_p = part.target_processor(prefix);
    std::size_t left_to_send = local_size;
    for (; left_to_send > 0 && first_p < comm.size(); ++first_p) {
        std::size_t nsend = std::min<std::size_t>(part.prefix_size(first_p) - prefix, left_to_send);
        send_counts[first_p] = nsend;
        left_to_send -= nsend;
        prefix += nsend;
    }
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    // TODO: accept iterators in mxx::all2all?
    mxx::all2allv(&(*begin), send_counts, &(*out), recv_counts, comm);
}


// TODO: remove code duplication below
/*
 * Re-distirbuted the vector into a perfect block decomposition.
 * (This invalidates all previous iterators)
 */
template <typename T>
std::vector<T> stable_block_decompose(const std::vector<T>& local_els, const mxx::comm& comm) {
    // if there's only one process, return a copy
    if (comm.size() == 1)
        return std::vector<T>(local_els);

    // get local size
    std::size_t local_size = local_els.size();
    // get prefix sum of size and total size
    std::size_t prefix = mxx::exscan(local_size, comm);
    std::size_t total_size = mxx::allreduce(local_size, comm);
    partition::block_decomposition<std::size_t> part(total_size, comm.size(), comm.rank());

    // allocate output
    std::vector<T> buffer(part.local_size());
    // calculate send counts
    // TODO: this is the same as above -> refactor
    // TODO: arbit_decomp class with takes local size in constructor
    //       and saves total and prefix size
    std::vector<size_t> send_counts(comm.size(), 0);
    int first_p = part.target_processor(prefix);
    std::size_t left_to_send = local_size;
    for (; left_to_send > 0 && first_p < comm.size(); ++first_p) {
        std::size_t nsend = std::min<std::size_t>(part.prefix_size(first_p) - prefix, left_to_send);
        send_counts[first_p] = nsend;
        left_to_send -= nsend;
        prefix += nsend;
    }
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    // TODO: accept iterators in mxx::all2all?
    mxx::all2allv(&local_els[0], send_counts, &buffer[0], recv_counts, comm);
    return buffer;
}

std::vector<size_t> surplus_send_pairing(std::vector<long long>& surpluses, int p, int rank, bool send_deficit = true)
{
    // calculate the send and receive counts by a linear scan over
    // the surpluses, using a queue to keep track of all surpluses
    std::vector<size_t> send_counts(p, 0);
    std::queue<std::pair<int, long long> > fifo;
    for (int i = 0; i < p; ++i)
    {
        if (surpluses[i] == 0)
            continue;
        if (fifo.empty()) {
            fifo.push(std::make_pair(i, surpluses[i]));
        } else if (surpluses[i] > 0) {
            if (fifo.front().second > 0) {
                fifo.push(std::make_pair(i, surpluses[i]));
            } else {
                while (surpluses[i] > 0 && !fifo.empty())
                {
                    long long min = std::min(surpluses[i], -fifo.front().second);
                    int j = fifo.front().first;
                    surpluses[i] -= min;
                    fifo.front().second += min;
                    if (fifo.front().second == 0)
                        fifo.pop();
                    // these processors communicate!
                    if (rank == i)
                        send_counts[j] += min;
                    else if (rank == j && send_deficit)
                        send_counts[i] += min;
                }
                if (surpluses[i] > 0)
                    fifo.push(std::make_pair(i, surpluses[i]));
            }
        } else if (surpluses[i] < 0) {
            if (fifo.front().second < 0) {
                fifo.push(std::make_pair(i, surpluses[i]));
            } else {
                while (surpluses[i] < 0 && !fifo.empty())
                {
                    long long min = std::min(-surpluses[i], fifo.front().second);
                    int j = fifo.front().first;
                    surpluses[i] += min;
                    fifo.front().second -= min;
                    if (fifo.front().second == 0)
                        fifo.pop();
                    // these processors communicate!
                    if (rank == i && send_deficit)
                        send_counts[j] += min;
                    else if (rank == j)
                        send_counts[i] += min;
                }
                if (surpluses[i] < 0)
                    fifo.push(std::make_pair(i, surpluses[i]));
            }
        }
    }
    assert(fifo.empty());

    return send_counts;
}

// non-stable version of the function above
// this is in-place and resizes the given vector internally!
template <typename T>
void block_decompose(std::vector<T>& local_els, const mxx::comm& comm) {
    // if single process, don't do anything
    if (comm.size() == 1)
        return;

    // get sizes
    long long local_size = local_els.size();
    long long total_size = mxx::allreduce(local_size, comm);
    partition::block_decomposition<std::size_t> part(total_size, comm.size(), comm.rank());
    long long surplus = local_size - (long long)part.local_size();
    std::vector<long long> surpluses = mxx::allgather(surplus, comm);
    assert(std::accumulate(surpluses.begin(), surpluses.end(), 0) == 0);

    // get send counts
    std::vector<size_t> send_counts = surplus_send_pairing(surpluses, comm.size(), comm.rank(), false);

#ifndef NDEBUG
    std::size_t send_size = std::accumulate(send_counts.begin(), send_counts.end(), 0);
    assert(surplus <= 0 || send_size == surplus);
#endif

    // allocate result
    std::vector<T> buffer;
    // send from the surplus, receive into buffer
    if (surplus > 0) {
        buffer = mxx::all2allv(&local_els[local_els.size()-surplus], send_counts, comm);
        local_els.resize(local_size - surplus);
        MXX_ASSERT(buffer.size() == 0);
    } else {
        MXX_ASSERT(send_size == 0);
        buffer = mxx::all2allv(&local_els[0], send_counts, comm);
        MXX_ASSERT(buffer.size() == -surplus);
        if (buffer.size() > 0) {
            local_els.resize(local_size - surplus);
            std::copy(buffer.begin(), buffer.end(), local_els.end() + surplus);
        }
    }
}
/**
 * @brief Redistributes elements from the given decomposition across processors
 *        into the decomposition given by the requested local_size
 */
template<typename _InIterator, typename _OutIterator>
void redo_arbit_decomposition(_InIterator begin, _InIterator end, _OutIterator out, std::size_t new_local_size, const mxx::comm& comm) {
    // get local size
    std::size_t local_size = std::distance(begin, end);

    // if single process, simply copy to output
    if (comm.size() == 1) {
        MXX_ASSERT(local_size == new_local_size);
        std::copy(begin, end, out);
        return;
    }

    // get prefix sum of size and total size
    size_t total_size = mxx::allreduce(local_size, comm);
    size_t prefix = mxx::exscan(local_size, comm);
    if (comm.rank() == 0)
        prefix = 0;

#if MEASURE_LOAD_BALANCE
    size_t min = mxx::reduce(local_size, 0, mxx::min<size_t>(), comm);
    size_t max = mxx::reduce(local_size, 0, mxx::max<size_t>(), comm);
    size_t min_new = mxx::reduce(new_local_size, 0, mxx::min<size_t>(), comm);
    size_t max_new = mxx::reduce(new_local_size, 0, mxx::max<size_t>(), comm);
    if(comm.rank() == 0)
      std::cerr << " Decomposition: old [" << min << "," << max << "], new= [" << min_new << "," << max_new << "], for n=" << total_size << " fair decomposition: " << total_size / comm.size() << std::endl;

    std::vector<std::size_t> toReceive = mxx::gather(new_local_size, 0, comm);
    if(comm.rank() == 0)
      std::cerr << toReceive << std::endl;
#endif

    // get the new local sizes from all processors
    // NOTE: this all-gather is what makes the arbitrary decomposition worse
    // in terms of complexity than when assuming a block decomposition
    std::vector<std::size_t> new_local_sizes = mxx::allgather(new_local_size, comm);
    MXX_ASSERT(std::accumulate(new_local_sizes.begin(), new_local_sizes.end(), 0ull) == total_size);

    // calculate where to send elements
    std::vector<size_t> send_counts(comm.size(), 0);
    int first_p;
    std::size_t new_prefix = 0;
    for (first_p = 0; first_p < comm.size()-1; ++first_p)
    {
        // find processor for which the prefix sum exceeds mine
        // i have to send to the previous
        if (new_prefix + new_local_sizes[first_p] > prefix)
            break;
        new_prefix += new_local_sizes[first_p];
    }

    //= block_partition_target_processor(total_size, p, prefix);
    std::size_t left_to_send = local_size;
    for (; left_to_send > 0 && first_p < comm.size(); ++first_p)
    {
        // make the `new` prefix inclusive (is an exlcusive prefix prior)
        new_prefix += new_local_sizes[first_p];
        // send as many elements to the current processor as it needs to fill
        // up, but at most as many as I have left
        std::size_t nsend = std::min<std::size_t>(new_prefix - prefix, left_to_send);
        send_counts[first_p] = nsend;
        // update the number of elements i have left (`left_to_send`) and
        // at which global index they start `prefix`
        left_to_send -= nsend;
        prefix += nsend;
    }

    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    // TODO: all2allv for iterators
    mxx::all2allv(&(*begin), send_counts, &(*out), recv_counts, comm);
}

// function for block decompsing vector of two partitions (for equal
// distributing of two halves)
// this one is stable for both halves.
template<typename _Iterator>
_Iterator stable_block_decompose_partitions(_Iterator begin, _Iterator mid, _Iterator end, const mxx::comm& comm)
{
    typedef typename std::iterator_traits<_Iterator>::value_type T;
    // return same sequence if there's a single process
    if (comm.size() == 1)
        return mid;

    // get sizes
    std::size_t left_local_size = std::distance(begin, mid);
    std::size_t right_local_size = std::distance(mid, end);
    // TODO: use array of 2 (single reduction!)
    std::size_t left_size = mxx::allreduce(left_local_size, comm);
    std::size_t right_size = mxx::allreduce(right_local_size, comm);
    partition::block_decomposition<std::size_t> part(left_size+right_size, comm.size(), comm.rank());
    partition::block_decomposition<std::size_t> left_part(left_size, comm.size(), comm.rank());

    // shuffle into buffer
    std::vector<T> buffer(part.local_size());
    redo_block_decomposition(begin, mid, buffer.begin(), comm);
    redo_arbit_decomposition(mid, end, buffer.begin()+left_part.local_size(), part.local_size() - left_part.local_size(), comm);

    // copy back
    std::copy(buffer.begin(), buffer.end(), begin);
    return begin + left_part.local_size();
}

// non-stable version (much faster, since data exchange is only for unequal parts)
template<typename _Iterator>
_Iterator block_decompose_partitions(_Iterator begin, _Iterator mid, _Iterator end, const mxx::comm& comm)
{
    typedef typename std::iterator_traits<_Iterator>::value_type T;
    // return same sequence if there's a single process
    if (comm.size() == 1)
        return mid;
    // get sizes
    long long left_local_size = std::distance(begin, mid);
    long long left_size = mxx::allreduce(left_local_size, comm);
    partition::block_decomposition<std::size_t> left_part(left_size, comm.size(), comm.rank());
    long long surplus = left_local_size - (long long)left_part.local_size();
    bool fits = end-begin >= left_part.local_size();
    bool all_fits = mxx::all_of(fits, comm);
    if (!all_fits)
        return mid;
    std::vector<long long> surpluses = mxx::allgather(surplus, comm);

    MXX_ASSERT(std::accumulate(surpluses.begin(), surpluses.end(), 0) == 0);

    // get send counts
    std::vector<size_t> send_counts = surplus_send_pairing(surpluses, comm.size(), comm.rank(), true);

    std::size_t send_size = std::accumulate(send_counts.begin(), send_counts.end(), 0);
    std::vector<T> buffer;
    if (send_size > 0)
        buffer.resize(send_size);

    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    // assert send and receive size are the same!
    for (int i = 0; i < comm.size(); ++i) {
        MXX_ASSERT(send_counts[i] == recv_counts[i]);
    }

    // send from the surplus, receive into buffer
    if (surplus > 0) {
        mxx::all2allv(&(*(mid - surplus)), send_counts, &buffer[0], recv_counts, comm);
        std::copy(buffer.begin(), buffer.end(), mid-surplus);
    } else if (surplus < 0) {
        mxx::all2allv(&(*mid), send_counts, &buffer[0], recv_counts, comm);
        std::copy(buffer.begin(), buffer.end(), mid);
    } else {
        MXX_ASSERT(send_size == 0);
        mxx::all2allv((T*)nullptr, send_counts, (T*)nullptr, recv_counts, comm);
    }

    // copy back
    return mid - surplus;
}
} // namespace mxx

#endif // MXX_DISTRIBUTION_HPP
