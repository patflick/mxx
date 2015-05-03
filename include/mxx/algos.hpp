/**
 * @file    algos.hpp
 * @author  Nagakishore Jammula <njammula3@mail.gatech.edu>
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements some common sequential algorithms
 *
 * Copyright (c) TODO
 *
 * TODO add Licence
 */

#ifndef HPC_ALGOS_H
#define HPC_ALGOS_H

#include <cstdlib>
#include <iterator>


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


/**
 * @brief Returns whether the given input range is in sorted order.
 *
 * @param begin An iterator to the begin of the sequence.
 * @param end   An iterator to the end of the sequence.
 *
 * @return  Whether the input sequence is sorted.
 */
template <typename Iterator>
bool is_sorted(Iterator begin, Iterator end)
{
    if (begin == end)
        return true;
    typename std::iterator_traits<Iterator>::value_type last = *(begin++);
    while (begin != end)
    {
        if (last > *begin)
            return false;
        last = *(begin++);
    }
    return true;
}

/**
 * @brief Performs a balanced partitioning.
 *
 * This function internally does 3-way partitioning (i.e. partitions the
 * input sequence into elements of the three classes given by:  <, ==, >
 *
 * After partioning, the elements that are in the `==` class are put into the
 * class of `<` or `>`.
 *
 * This procedure ensures that if the pivot is part of the sequence, then
 * the returned left (`<`) and right (`>`) sequences are both at least one
 * element in length, i.e. neither is of length zero.
 *
 * @param begin An iterator to the beginning of the sequence.
 * @param end   An iterator to the end of the sequence.
 * @param pivot A value to be used as pivot.
 *
 * @return An iterator pointing to the first element of the right (`>`)
 *         sequence.
 */
template<typename Iterator, typename T>
Iterator balanced_partitioning(Iterator begin, Iterator end, T pivot)
{
    if (begin == end)
        return begin;

    // do 3-way partitioning and then balance the result
    Iterator eq_it = begin;
    Iterator le_it = begin;
    Iterator ge_it = end - 1;

    while (true)
    {
        while (le_it < ge_it && *le_it < pivot) ++le_it;
        while (le_it < ge_it && *ge_it > pivot) --ge_it;

        if (le_it == ge_it)
        {
            if (*le_it == pivot)
            {
                std::swap(*(le_it++), *(eq_it++));
            }
            else if (*le_it < pivot)
            {
                le_it++;
            }
            break;
        }

        if (le_it > ge_it)
            break;

        if (*le_it == pivot)
        {
            std::swap(*(le_it++), *(eq_it++));
        } else if (*ge_it == pivot)
        {
            std::swap(*(ge_it--), *(le_it));
            std::swap(*(le_it++), *(eq_it++));
        }
        else
        {
            std::swap(*(le_it++), *(ge_it--));
        }
    }


    // if there are pivots, put them to the smaller side
    if (eq_it != begin)
    {
        typedef typename std::iterator_traits<Iterator>::difference_type diff_t;
        diff_t left_size = le_it - eq_it;
        diff_t right_size = end - le_it;

        // if left is bigger, put to right, otherwise just leave where they are
        if (left_size > right_size)
        {
            // put elements to right
            while (eq_it > begin)
            {
                std::swap(*(--le_it), *(--eq_it));
            }
        }
    }
    // no pivots, therefore simply return the partitioned sequence
    return le_it;
}

#endif // HPC_ALGOS_H
