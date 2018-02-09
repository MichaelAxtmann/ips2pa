/******************************************************************************
 * ips2pa/partitioning.hpp
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

#include <atomic>
#include <tuple>
#include <utility>

#include "memory.hpp"
#include "utils.hpp"

#include "ips2pa_fwd.hpp"
#include "sampling.hpp"
#include "local_classification.hpp"
#include "block_permutation.hpp"
#include "cleanup_margins.hpp"

namespace ips2pa {
namespace detail {

template <class Cfg>
template <bool kIsParallel>
bool Sorter<Cfg>::buildSortingClassifier(const iterator begin,
                                         const iterator end,
                                         SharedData* const shared) {
    // Sampling
    bool use_equal_buckets = false;
    if (!kIsParallel) {
        auto& classifier = local_.classifier;
        int log_buckets = 0;
        std::tie(num_buckets_, log_buckets, use_equal_buckets) = selectSplitters(begin, end, classifier);
        // Build the tree
        classifier.build(log_buckets);
        classifier_ = &classifier;
    } else {
        shared->sync.single([&] {
                auto& classifier = shared->classifier;// local_.classifier;
                int log_buckets = 0;
                std::tie(num_buckets_, log_buckets, use_equal_buckets) = selectSplitters(begin, end, classifier);
                // Build the tree
                classifier.build(log_buckets);
                classifier_ = &classifier;
                shared->num_buckets = num_buckets_;
                shared->use_equal_buckets = use_equal_buckets;
            });
        num_buckets_ = shared->num_buckets;
        use_equal_buckets = shared->use_equal_buckets;
    }
    return use_equal_buckets;
}

template <class Cfg>
template <bool kIsParallel>
typename Cfg::difference_type Sorter<Cfg>::buildPartitioningClassifier(const iterator s_begin,
                                                                       const iterator s_end,
                                                                       diff_t n,
                                                                       diff_t num_splitter,
                                                                       bool use_equal_buckets,
                                                                       SharedData* const shared) {
    diff_t step = 0;
    if (!kIsParallel) {
        auto& classifier = local_.classifier;
        int log_buckets = 0;
        std::tie(num_buckets_, log_buckets, step) = selectPartitionerSplitters(s_begin, s_end, n, num_splitter, classifier, use_equal_buckets);
        // Build the tree
        classifier.build(log_buckets);
        classifier_ = &classifier;
    } else {
        shared->sync.single([&] {
                auto& classifier = shared_->classifier;
                int log_buckets = 0;
                std::tie(num_buckets_, log_buckets, step) = selectPartitionerSplitters(s_begin, s_end, n, num_splitter, classifier, use_equal_buckets);
                // Build the tree
                classifier.build(log_buckets);
                classifier_ = &classifier;
                shared->num_buckets = num_buckets_;
                shared->use_equal_buckets = use_equal_buckets;
            });
        num_buckets_ = shared->num_buckets;
    }
    return step;
}

/**
 * Main partitioning function.
 */
template <class Cfg>
template <bool kIsParallel, bool post_sort>
int Sorter<Cfg>::partition(const iterator begin, const iterator end,
                           diff_t* const bucket_start,
                           SharedData* const shared, const int my_id,
                           const int num_threads,
                           bool use_equal_buckets) {
    // Set parameters for this partitioning step
    // Must do this AFTER sampling, because sampling will recurse to sort splitters.
    this->shared_ = shared;
    this->classifier_ = kIsParallel ? &shared_->classifier : &local_.classifier;
    this->bucket_start_ = bucket_start;
    this->bucket_pointers_ = kIsParallel ? shared_->bucket_pointers : local_.bucket_pointers;
    this->overflow_ = nullptr;
    this->begin_ = begin;
    this->end_ = end;
    this->my_id_ = my_id;
    this->num_threads_ = num_threads;

    // Local Classification
    if (kIsParallel)
        parallelClassification(use_equal_buckets);
    else
        sequentialClassification(use_equal_buckets);

    // Compute which bucket can cause overflow
    const int overflow_bucket = computeOverflowBucket();

    // Block Permutation
    if (use_equal_buckets)
        permuteBlocks<true, kIsParallel>();
    else
        permuteBlocks<false, kIsParallel>();

    if (kIsParallel && overflow_)
        shared_->overflow = &local_.overflow;

    if (kIsParallel) shared_->sync.barrier();

    // Cleanup
    {
        if (kIsParallel) overflow_ = shared_->overflow;

        // Distribute buckets among threads
        const int num_buckets = num_buckets_;
        const int buckets_per_thread = (num_buckets + num_threads_ - 1) / num_threads_;
        int my_first_bucket = my_id_ * buckets_per_thread;
        int my_last_bucket = (my_id_ + 1) * buckets_per_thread;
        my_first_bucket = num_buckets < my_first_bucket ? num_buckets : my_first_bucket;
        my_last_bucket = num_buckets < my_last_bucket ? num_buckets : my_last_bucket;

        // Save excess elements at right end of stripe
        const auto in_swap_buffer = !kIsParallel
                                            ? std::pair<int, diff_t>(my_last_bucket, 0)
                                            : saveMargins(my_last_bucket);
        if (kIsParallel) shared_->sync.barrier();

        // Write remaining elements
        writeMargins<post_sort>(my_first_bucket, my_last_bucket, overflow_bucket,
                     in_swap_buffer.first, in_swap_buffer.second);
    }

    if (kIsParallel) shared_->sync.barrier();
    local_.reset();

    return this->num_buckets_;
}

}  // namespace detail
}  // namespace ips2pa
