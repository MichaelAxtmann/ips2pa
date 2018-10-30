/******************************************************************************
 * ips2pa/sampling.hpp
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

#include <iterator>
#include <random>
#include <utility>

#include "ips2pa_fwd.hpp"
#include "config.hpp"
#include "classifier.hpp"
#include "memory.hpp"

namespace ips2pa {
namespace detail {

/**
 * Selects a random sample in-place.
 */
template <class It, class RandomGen>
void selectSample(It begin, const It end,
                  typename std::iterator_traits<It>::difference_type num_samples,
                  RandomGen&& gen) {
    using std::swap;

    auto n = end - begin;
    while (num_samples--) {
        const auto i = std::uniform_int_distribution<typename std::iterator_traits<It>::difference_type>(0, --n)(gen);
        swap(*begin, begin[i]);
        ++begin;
    }
}

/**
 * Builds the classifer by selecting the splitters.
 */
template <class Cfg>
std::tuple<int, int, bool> Sorter<Cfg>::selectSplitters(const iterator begin,
                                                        const iterator end,
                                                        Classifier& classifier) {
    const auto n = end - begin;
    int log_buckets = Cfg::logBuckets(n);
    int num_buckets = 1 << log_buckets;
    const auto step = std::max<diff_t>(1, Cfg::oversamplingFactor(n));
    const auto num_samples = step * num_buckets - 1;

    // Select the sample
    detail::selectSample(begin, end, num_samples, local_.random_generator);

    // Sort the sample
    sequentialSort(begin, begin + num_samples);
    auto splitter = begin + step - 1;
    auto sorted_splitters = classifier.getSortedSplitters();
    const auto comp = classifier.getComparator();

    // Choose the splitters
    IPS2PA_ASSUME_NOT(sorted_splitters == nullptr);
    new (sorted_splitters) typename Cfg::value_type(*splitter);
    for (int i = 2; i < num_buckets; ++i) {
        splitter += step;
        // Skip duplicates
        if (comp(*sorted_splitters, *splitter)) {
            IPS2PA_ASSUME_NOT(sorted_splitters + 1 == nullptr);
            new (++sorted_splitters) typename Cfg::value_type(*splitter);
        }
    }

    // Check for duplicate splitters
    const auto diff_splitters = sorted_splitters - classifier.getSortedSplitters() + 1;
    const bool use_equal_buckets = Cfg::kAllowEqualBuckets
        && (num_buckets - 1 - diff_splitters >= Cfg::kEqualBucketsThreshold
            || diff_splitters == 1);

    // Fill the array to the next power of two
    log_buckets = log2(diff_splitters) + 1;
    num_buckets = 1 << log_buckets;
    for (int i = diff_splitters + 1; i < num_buckets; ++i) {
        IPS2PA_ASSUME_NOT(sorted_splitters + 1 == nullptr);
        new (++sorted_splitters) typename Cfg::value_type(*splitter);
    }

    const int used_buckets = num_buckets * (1 + use_equal_buckets);
    return std::tuple<int, int, bool>{used_buckets, log_buckets, use_equal_buckets};
}

/**
 * Builds the partitioning classifer by collecting splitters equidistantly.
 */
template <class Cfg>
std::tuple<int, int, typename Cfg::difference_type> Sorter<Cfg>::selectPartitionerSplitters(
    const iterator s_begin,
    diff_t num_in_splitter,
    Classifier& classifier,
    bool use_equal_buckets) {

    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;
    const int log_buckets = Cfg::logBucketsPartitioner(num_in_splitter + 1);
    const int num_buckets = 1 << log_buckets;

    // num_buckets may be larger than b.
    const auto step = std::max<diff_t>((num_in_splitter + num_buckets) / num_buckets, 1);
    auto splitter = s_begin + step - 1;
    auto sorted_splitters = classifier.getSortedSplitters();
    
    // Choose the splitters
    IPS2PA_ASSUME_NOT(sorted_splitters == nullptr);
    new (sorted_splitters) value_type(*splitter);
    for (int i = 1; i < num_in_splitter / step; ++i) {
        splitter += step;
        IPS2PA_ASSUME_NOT(sorted_splitters + 1 == nullptr);
        new (++sorted_splitters) value_type(*splitter);
    }

    // Fill the array to the next power of two
    for (int i = num_in_splitter / step; i < num_buckets - 1; ++i) {
        IPS2PA_ASSUME_NOT(sorted_splitters + 1 == nullptr);
        new (++sorted_splitters) value_type(*splitter);
    }

    const int used_buckets = num_buckets * (1 + use_equal_buckets);
    return std::tuple<int, int, diff_t>{used_buckets, log_buckets, step};
}

}  // namespace detail
}  // namespace ips2pa
