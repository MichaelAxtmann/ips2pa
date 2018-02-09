/******************************************************************************
 * ips2pa/base_case.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 *
 ******************************************************************************
 * BSD 2-Clause License
 *
 * Copyright © 2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <assert.h>

#include "ips2pa_fwd.hpp"
#include "utils.hpp"

namespace ips2pa {
namespace detail {

/**
 * Insertion sort.
 */
template <class It, class Comp>
void insertionSort(const It begin, const It end, Comp comp) {
    IPS2PA_ASSUME_NOT(begin >= end);

    for (It it = begin + 1; it < end; ++it) {
        auto val = std::move(*it);
        if (comp(val, *begin)) {
            std::move_backward(begin, it, it + 1);
            *begin = std::move(val);
        } else {
            auto cur = it;
            for (auto next = it - 1; comp(val, *next); --next) {
                *cur = std::move(*next);
                cur = next;
            }
            *cur = std::move(val);
        }
    }
}

/**
 * Wrapper for base case sorter, for easier swapping.
 */
template <class It, class Comp>
inline void baseCaseSort(It begin, It end, Comp&& comp) {
    if (begin == end) return;
    detail::insertionSort(std::move(begin), std::move(end), std::forward<Comp>(comp));
}

/**
 * Partitions the input into buckets by merging.
 */
template <class It, class Comp>
inline void mergeInSortedBuckets(const It begin,
                                 const It end,
                                 const It s_begin,
                                 const It s_end,
                                 size_t* bs_begin,
                                 bool use_equal_buckets,
                                 Comp&& comp) {
    auto prev_el_it = begin;
    auto el_it = begin;
    auto s_it = s_begin;
    const size_t num_splitter = s_end - s_begin;
    if (use_equal_buckets) {
        for (size_t i = 0; i != num_splitter; ++i) {
            // Less bucket
            while (el_it != end && comp(*el_it, *s_it)) {
                ++el_it;
            }
            bs_begin[2 * i] = el_it - prev_el_it;
            prev_el_it = el_it;

            // Equal bucket
            while (el_it != end && !comp(*s_it, *el_it)) {
                ++el_it;
            }
            bs_begin[2 * i + 1] = el_it - prev_el_it;
            prev_el_it = el_it;
            
            ++s_it;
        }
    } else {
        for (size_t i = 0; i != num_splitter; ++i) {
            // Less bucket
            while (el_it != end && !comp(*s_it, *el_it)) {
                ++el_it;
            }
            bs_begin[i] = el_it - prev_el_it;
            prev_el_it = el_it;
             
            ++s_it;
        }
    }

    bs_begin[(1 + use_equal_buckets) * num_splitter] = end - el_it;
    return;
}

}  // namespace detail
}  // namespace ips2pa
