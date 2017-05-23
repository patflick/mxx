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
 * @file    paritition.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the partition API and different data partitions, most
 *          notably the block decomposition.
 */

#ifndef MXX_PARTITION_HPP
#define MXX_PARTITION_HPP

#include <vector>
#include <cstddef>
#include <assert.h>

#include "comm.hpp"

namespace mxx
{
namespace partition
{

// We want inlined non virtual functions for the partition. We thus need to use
// templating when these classes are used and no real inheritance to enforce
// the API.
#if 0
template <typename index_t>
class block_decomposition
{
public:
    block_decomposition() : n(0), p(0), rank(0) {}

    /// Constructor (no default construction)
    block_decomposition(index_t n, int p, int rank)
        : n(n), p(p), rank(rank)
    {
    }

    index_t local_size()
    {
        return n/p + ((static_cast<index_t>(rank) < (n % static_cast<index_t>(p))) ? 1 : 0);
    }

    inline index_t global_size() const {
        return n;
    }

    index_t local_size(int rank)
    {
        return n/p + ((static_cast<index_t>(rank) < (n % static_cast<index_t>(p))) ? 1 : 0);
    }

    index_t prefix_size()
    {
        return (n/p)*(rank+1) + std::min<index_t>(n % p, rank + 1);
    }

    index_t prefix_size(int rank)
    {
        return (n/p)*(rank+1) + std::min<index_t>(n % p, rank + 1);
    }

    index_t excl_prefix_size()
    {
        return (n/p)*rank + std::min<index_t>(n % p, rank);
    }

    index_t excl_prefix_size(int rank)
    {
        return (n/p)*rank + std::min<index_t>(n % p, rank);
    }

    // which processor the element with the given global index belongs to
    int target_processor(index_t global_index)
    {
        if (global_index < ((n/p)+1)*(n % p))
        {
            // a_i is within the first n % p processors
            return global_index/((n/p)+1);
        }
        else
        {
            return n%p + (global_index - ((n/p)+1)*(n % p))/(n/p);
        }
    }

    /// Destructor
    virtual ~block_decomposition () {}
private:
    /* data */
    /// Number of elements
    index_t n;
    /// Number of processors
    int p;
    /// Processor rank
    int rank;
};


template <typename index_t>
class block_decomposition_buffered
{
public:
    block_decomposition_buffered() {}

    block_decomposition_buffered(index_t n, int p, int rank)
        : n(n), p(p), rank(rank), div(n / p), mod(n % p),
          loc_size(div + (static_cast<index_t>(rank) < mod ? 1 : 0)),
          prefix(div*rank + std::min<index_t>(mod, rank)),
          div1mod((div+1)*mod)
    {
    }

    block_decomposition_buffered(const block_decomposition_buffered& other)
        : n(other.n), p(other.p), rank(other.rank), div(other.div),
        mod(other.mod), loc_size(other.loc_size), prefix(other.prefix),
        div1mod(other.div1mod) {}

    block_decomposition_buffered& operator=(const block_decomposition_buffered& other) = default;

    index_t local_size()
    {
        return loc_size;
    }

    index_t local_size(int rank)
    {
        return div + (static_cast<index_t>(rank) < mod ? 1 : 0);
    }

    inline index_t global_size() const {
        return n;
    }

    index_t prefix_size()
    {
        return prefix + loc_size;
    }

    index_t prefix_size(int rank)
    {
        return div*(rank+1) + std::min<index_t>(mod, rank + 1);
    }

    index_t excl_prefix_size()
    {
        return prefix;
    }

    index_t excl_prefix_size(int rank)
    {
        return div*rank + std::min<index_t>(mod, rank);
    }

    // which processor the element with the given global index belongs to
    int target_processor(index_t global_index)
    {
        if (global_index < div1mod)
        {
            // a_i is within the first n % p processors
            return global_index/(div+1);
        }
        else
        {
            return mod + (global_index - div1mod)/div;
        }
    }

    virtual ~block_decomposition_buffered () {}
private:
    /* data */
    /// Number of elements
    index_t n;
    /// Number of processors
    int p;
    /// Processor rank
    int rank;

    // derived/buffered values (for faster computation of results)
    index_t div; // = n/p
    index_t mod; // = n%p
    // local size (number of local elements)
    index_t loc_size;
    // the exclusive prefix (number of elements on previous processors)
    index_t prefix;
    /// number of elements on processors with one more element
    index_t div1mod; // = (n/p + 1)*(n % p)
};
#endif
} // namespace partition


// rough interface for distribution of data elements over the processors of a
// communicator
class dist_base {
protected:
    unsigned int m_comm_size, m_comm_rank;
    size_t m_global_size;

    dist_base() : m_comm_size(0), m_comm_rank(0), m_global_size(0) {}

    dist_base(size_t global_size, const mxx::comm& comm) :
        m_comm_size(comm.size()), m_comm_rank(comm.rank()), m_global_size(global_size) {
    }

    dist_base(size_t global_size, int comm_size, int comm_rank) :
        m_comm_size(comm_size), m_comm_rank(comm_rank), m_global_size(global_size) {
    }

public:
    inline size_t global_size() const {
        return m_global_size;
    }

    inline int comm_size() const {
        return m_comm_size;
    }

    inline int comm_rank() const {
        return m_comm_rank;
    }

    /* need to be implemented by all deriving classes
    // simple size and prefix queries
    inline size_t local_size() const;
    inline size_t local_size(int rank) const;
    inline size_t global_size() const;
    inline size_t eprefix_size() const;
    inline size_t iprefix_size() const;
    inline size_t eprefix_size(int rank) const;
    inline size_t iprefix_size(int rank) const;

    // rank and gidx translations
    inline int    rank_of(size_t gidx) const;
    inline size_t lidx_of(size_t gidx) const;
    inline size_t gidx_of(int rank, size_t lidx) const;
    */
};

/*
 * TODO:
 * move simple bare queries to easier base class:
 * local_size(), global_size(), eprefix_size(), iprefix_size()
 */


class blk_dist_buf : public dist_base {
public:
    using dist_base::global_size;

    ~blk_dist_buf () {}

    /// collective allreduce for global size
    /*
    blk_dist_buf(const mxx::comm& comm, size_t local_size)
        : dist_base(comm, local_size),
          n(mxx::allreduce(local_size, comm)),
          div(n / m_comm_size), mod(n % m_comm_size),
          prefix(div*m_comm_rank + std::min<size_t>(mod, m_comm_rank)),
          div1mod((div+1)*mod)
    {
        assert(local_size == div + (m_comm_rank < mod ? 1 : 0));
    }
    */

    blk_dist_buf() = default;

    blk_dist_buf(size_t global_size, unsigned int comm_size, unsigned int comm_rank)
        : dist_base(global_size, comm_size, comm_rank),
          div(global_size / comm_size), mod(global_size % comm_size),
          m_local_size(div + (comm_rank < mod ? 1 : 0)),
          m_prefix(div*m_comm_rank + std::min<size_t>(mod, m_comm_rank)),
          div1mod((div+1)*mod) {
    }

    blk_dist_buf(size_t global_size, const mxx::comm& comm)
        : blk_dist_buf(global_size, comm.size(), comm.rank()) {}

    // default copy and move
    blk_dist_buf(blk_dist_buf&&) = default;
    blk_dist_buf(const blk_dist_buf&) = default;
    blk_dist_buf& operator=(const blk_dist_buf&) = default;
    blk_dist_buf& operator=(blk_dist_buf&&) = default;

    inline size_t local_size() const {
        return m_local_size;
    }

    inline size_t local_size(unsigned int rank) const {
        return div + (rank < mod ? 1 : 0);
    }

    inline size_t iprefix_size() const {
        return m_prefix + m_local_size;
    }

    inline size_t iprefix_size(unsigned int rank) const {
        return div*(rank+1) + std::min<size_t>(mod, rank + 1);
    }

    inline size_t eprefix_size() const {
        return m_prefix;
    }

    inline size_t eprefix_size(unsigned int rank) const {
        return div*rank + std::min<size_t>(mod, rank);
    }

    // which processor the element with the given global index belongs to
    inline int rank_of(size_t gidx) const {
        if (gidx < div1mod) {
            // a_i is within the first n % p processors
            return gidx/(div+1);
        } else {
            return mod + (gidx - div1mod)/div;
        }
    }

    inline size_t lidx_of(size_t gidx) const {
        return gidx - eprefix_size(rank_of(gidx));
    }

    inline size_t gidx_of(int rank, size_t lidx) const {
        return eprefix_size(rank) + lidx;
    }

private:
    /* data */
    // derived/buffered values (for faster computation of results)
    size_t div; // = n/p
    size_t mod; // = n%p
    // local size (number of local elements)
    size_t m_local_size;
    // the exclusive prefix (number of elements on previous processors)
    size_t m_prefix;
    /// number of elements on processors with one more element
    size_t div1mod; // = (n/p + 1)*(n % p)
};


using blk_dist = blk_dist_buf;


// simplified block distr: equal number of elements on each processor:
// exactly n/p (e.g.: the required input to bitonic sort)
// TODO: adapt and finish this:
/*
class eq_dist : public dist_base {
public:
    eq_dist(const mxx::comm& comm, size_t local_size) : dist_base(comm, local_size) {}

    inline size_t local_size(int) {
        return dist_base::local_size();
    }

    inline size_t global_size() const {
        return dist_base::local_size() * comm_size();
    }

    inline size_t eprefix() const {
        return m_local_size * m_comm_rank;
    }

    inline size_t iprefix() const {
        return m_local_size * (m_comm_rank + 1);
    }

    inline size_t eprefix(int rank) const {
        return m_local_size * rank;
    }

    inline size_t iprefix(int rank) const {
        return m_local_size * (rank + 1);
    }

    inline int    rank_of(size_t gidx) const {
        return gidx / m_local_size;
    }

    inline size_t lidx_of(size_t gidx) const {
        return gidx % m_local_size;
    }

    inline size_t gidx_of(int rank, size_t lidx) const {
        return m_local_size * rank + lidx;
    }
};
*/

} // namespace mxx

#endif // MXX_PARTITION_HPP
