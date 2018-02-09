/******************************************************************************
 * ips2pa/sequential.hpp
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

#include <utility>

#include "ips2pa_fwd.hpp"
#include "base_case.hpp"
#include "memory.hpp"
#include "partitioning.hpp"
#include <cstring>


namespace ips2pa {
namespace detail {

/**
 * Recursive entry point for sequential algorithm.
 */
template <class Cfg>
void Sorter<Cfg>::sequentialSort(const iterator begin, const iterator end) {
    // Check for base case
    const auto n = end - begin;
    if (n <= 2 * Cfg::kBaseCaseSize) {
        detail::baseCaseSort(begin, end, local_.classifier.getComparator());
        return;
    }
    diff_t bucket_start[Cfg::kMaxBuckets + 1];

    // Do the partitioning
    const bool use_equal_buckets = buildSortingClassifier<false>(begin, end, nullptr);
    const int num_buckets = partition<false, true>(begin, end, bucket_start, nullptr, 0, 1,
                                      use_equal_buckets);

    // Final base case is executed in cleanup step, so we're done here
    if (n <= Cfg::kSingleLevelThreshold) {
        return;
    }

    // Recurse
    for (int i = 0; i < num_buckets; i += 1 + use_equal_buckets) {
        const auto start = bucket_start[i];
        const auto stop = bucket_start[i + 1];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
            sequentialSort(begin + start, begin + stop);
    }
    if (use_equal_buckets) {
        const auto start = bucket_start[num_buckets - 1];
        const auto stop = bucket_start[num_buckets];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
            sequentialSort(begin + start, begin + stop);
    }
}

/**
 * Wrapper for base case partitioner, for easier swapping.
 */
template <class Cfg>
void Sorter<Cfg>::partitionByMerging(const iterator begin,
                              const iterator end,
                              const iterator s_begin,
                              const iterator s_end,
                              size_t* bs_begin,
                              bool use_equal_buckets) {
    sequentialSort(begin, end);
    detail::mergeInSortedBuckets(begin, end,
                                 s_begin, s_end,
                                 bs_begin, use_equal_buckets,
                                 local_.classifier.getComparator());
}

/**
 * Recursive entry point for sequential partitioning algorithm.
 */
template <class Cfg>
void Sorter<Cfg>::sequentialPartition(const iterator begin,
                                      const iterator end,
                                      const iterator s_begin,
                                      const iterator s_end,
                                      size_t* b_begin,
                                      bool use_equal_buckets) {
    // Check for base case
    const auto n = end - begin;
    const auto num_splitters = s_end - s_begin;
    const auto num_buckets = num_splitters + 1;
    // const auto& comp = local_.classifier.getComparator();
    if (num_splitters == 0 || n == 0) {
        *b_begin = n;
        return;
    } else if (n <= 2 * Cfg::kBaseCaseSize || n <= 2 * num_buckets) {
        partitionByMerging(begin, end,
                           s_begin, s_end,
                           b_begin, use_equal_buckets);
        return;
    }
    
    // /* If the number of input elements is large, we first partition the input without equal
    //  * buckets. If equal buckets are required, we then swap equal elements to the end 
    //  * of each bucket.
    //  */
    // if (use_equal_buckets) {
    //     sequentialPartition(begin, end,
    //                         s_begin, s_end,
    //                         b_begin,
    //                         false);
    //     const auto comp = local_.classifier.getComparator();
    //     size_t b_tmp[2 * (s_end - s_begin) + 1];
    //     auto part_begin = begin;
    //     for (std::ptrdiff_t i = 0; i != s_end - s_begin; ++i) {
    //         auto read_ptr = part_begin + b_begin[i];
    //         auto write_ptr = part_begin + b_begin[i];
    //         while (read_ptr != part_begin) {
    //             --read_ptr;
    //             if (!comp(*read_ptr, s_begin[i])) {
    //                 std::swap(*read_ptr, *(--write_ptr));
    //             }
    //         }
    //         b_tmp[2 * i] = write_ptr - part_begin;
    //         b_tmp[2 * i + 1] = b_begin[i] - b_tmp[2 * i];
    //         part_begin += b_begin[i];
    //     }
    //     b_tmp[2 * (s_end - s_begin)] = b_begin[s_end - s_begin];
    //     std::memcpy(b_begin, b_tmp, sizeof(size_t) * (2 * (s_end - s_begin) + 1));
    //     return;
    // }

    const diff_t step = buildPartitioningClassifier<false>(s_begin, s_end,
                                                 n, num_splitters, use_equal_buckets, nullptr);
    diff_t bucket_start[Cfg::kMaxBuckets + 1];
    const bool is_last_level = step == 1;
    partition<false, false>(begin, end, bucket_start, nullptr, 0, 1,
                     use_equal_buckets);

    // Recurse
    if (!use_equal_buckets) {
        bucket_start[num_splitters / step + 1] = bucket_start[num_buckets_];
        if (is_last_level) {
            for (diff_t idx = 0; idx != num_buckets; ++idx) {
                b_begin[idx] = bucket_start[idx + 1] - bucket_start[idx];
            }
            return;
        }
    
        for (diff_t i = 0; i < num_splitters / step + 1; ++i) {
            const auto rec_begin = begin + bucket_start[i];
            const auto rec_end = begin + bucket_start[i + 1];
            const auto s_rec_begin = s_begin + i * step;
            const auto s_rec_end = s_begin + std::min((i + 1) * step - 1, num_splitters);
            const auto b_rec_begin = b_begin + i * step;
            if (s_rec_begin == s_rec_end) {
                // Set size of last bucket if the last splitter we used in this recursion
                // is the last splitter overall.
                *b_rec_begin = bucket_start[num_splitters / step + 1] - bucket_start[num_splitters / step];
            } else if (rec_end - rec_begin >= Cfg::kUnrollClassifier) {
                sequentialPartition(rec_begin, rec_end,
                                      s_rec_begin, s_rec_end,
                                      b_rec_begin,
                                      use_equal_buckets);
            } else {
                partitionByMerging(rec_begin, rec_end,
                                   s_rec_begin, s_rec_end,
                                   b_rec_begin,
                                   use_equal_buckets);
            }
        }
    } else {
        bucket_start[2 * (num_splitters / step) + 1] = bucket_start[num_buckets_];
        if (is_last_level) {
            for (diff_t idx = 0; idx != 2 * num_splitters + 1; ++idx) {
                b_begin[idx] = bucket_start[idx + 1] - bucket_start[idx];
            }
            return;
        }
        for (diff_t i = 0; i < num_splitters / step + 1; ++i) {
            const auto rec_begin = begin + bucket_start[2 * i];
            const auto rec_end = begin + bucket_start[2 * i + 1];
            const auto s_rec_begin = s_begin + i * step;
            const auto s_rec_end = s_begin + std::min((i + 1) * step - 1, num_splitters);
            const auto b_rec_begin = b_begin + 2 * i * step;
            if (i > 0) {
                // Set size of equal bucket.
                *(b_rec_begin - 1) = bucket_start[2 * i] - bucket_start[2 * i - 1];
            }
            if (s_rec_begin == s_rec_end) {
                // Set size of last bucket if the last splitter we used in this recursion
                // is the last splitter overall.
                *b_rec_begin = bucket_start[2 * i + 1] - bucket_start[2 * i];
            } else if (rec_end - rec_begin >= Cfg::kUnrollClassifier) {
                sequentialPartition(rec_begin, rec_end,
                                    s_rec_begin, s_rec_end,
                                    b_rec_begin,
                                    use_equal_buckets);
            } else {
                partitionByMerging(rec_begin, rec_end,
                                   s_rec_begin, s_rec_end,
                                   b_rec_begin,
                                   use_equal_buckets);
            }
        }
    }
}

}  // namespace detail

/**
 * Reusable sequential sorter.
 */
template <class Cfg>
class SequentialSorter {
    using Sorter = detail::Sorter<Cfg>;
    using iterator = typename Cfg::iterator;

public:
    explicit SequentialSorter(typename Cfg::less comp, std::ptrdiff_t seed = detail::genSeed<Config<>>())
        : buffer_storage_(1)
        , local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage_.get(), seed) {}

    explicit SequentialSorter(typename Cfg::less comp,
                              char* buffer_storage,
                              std::ptrdiff_t seed = detail::genSeed<Config<>>())
        : local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage, seed) {}

    void operator()(iterator begin, iterator end) {
        Sorter(local_ptr_.get()).sequentialSort(std::move(begin), std::move(end));
    }

 private:
    typename Sorter::BufferStorage buffer_storage_;
    detail::AlignedPtr<typename Sorter::LocalData> local_ptr_;
};

/**
 * Reusable sequential partitioner.
 */
template <class Cfg>
class SequentialPartitioner {
    using Partitioner = detail::Sorter<Cfg>;
    using iterator = typename Cfg::iterator;

 public:
    explicit SequentialPartitioner(typename Cfg::less comp, std::ptrdiff_t seed)
            : buffer_storage_(1)
            , local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage_.get(), seed) {}

    explicit SequentialPartitioner(typename Cfg::less comp,
                                   std::ptrdiff_t seed,
                                   char* buffer_storage)
        : local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage, seed) {}

    void operator()(const iterator begin,
                    const iterator end,
                    const iterator s_begin,
                    const iterator s_end,
                    size_t* b_begin,
                    bool use_equal_buckets) {
        Partitioner(local_ptr_.get()).sequentialPartition(std::move(begin), std::move(end),
                                                          std::move(s_begin), std::move(s_end),
                                                          b_begin,
                                                          use_equal_buckets);
    }

 private:
    typename Partitioner::BufferStorage buffer_storage_;
    detail::AlignedPtr<typename Partitioner::LocalData> local_ptr_;
};

}  // namespace ips2pa
