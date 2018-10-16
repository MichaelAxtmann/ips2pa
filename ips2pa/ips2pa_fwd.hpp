/******************************************************************************
 * ips2pa/ips2pa_fwd.hpp
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
#include <utility>

namespace ips2pa {
namespace detail {

template <class It, class Comp>
inline void baseCaseSort(It begin, It end, Comp&& comp);

template <class It, class Comp>
inline void mergeInSortedBuckets(It begin,
                              It end,
                              It b_begin,
                              It b_end,
                              size_t* bs_begin,
                              bool use_equal_buckets,
                              Comp&& comp);

inline constexpr unsigned long log2(unsigned long n);

template <class It, class RandomGen>
inline void selectSample(It begin, It end,
                         typename std::iterator_traits<It>::difference_type num_samples,
                         RandomGen&& gen);

template <class Cfg>
class Sorter {
 public:
    using iterator = typename Cfg::iterator;
    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;

    class BufferStorage;
    class Block;
    class Buffers;
    class BucketPointers;
    class Classifier;
    struct LocalData;
    struct SharedData;
    explicit Sorter(LocalData& local) : local_(local) {}

    void sequentialPartition(iterator begin,
                             iterator end,
                             iterator s_begin,
                             iterator s_end,
                             size_t* b_begin,
                             bool use_equal_buckets);

    void sequentialSort(iterator begin, iterator end);

#if defined(_REENTRANT) || defined(_OPENMP)
    template <class TaskSorter>
    void parallelPrimary(iterator begin, iterator end, SharedData& shared,
                         int num_threads, TaskSorter&& task_sorter);

    void parallelSecondary(SharedData& shared, int id, int num_threads);
#endif

 private:
    LocalData& local_;
    SharedData* shared_;
    Classifier* classifier_;

    diff_t* bucket_start_;
    BucketPointers* bucket_pointers_;
    Block* overflow_;

    iterator begin_;
    iterator end_;
    int num_buckets_;
    int my_id_;
    int num_threads_;

    static inline int computeLogBuckets(diff_t n);

    std::tuple<int, int, bool> selectSplitters(iterator begin,
                                               iterator end,
                                               Classifier& classifier);

    std::tuple<int, int, diff_t> selectPartitionerSplitters(iterator s_begin,
                                                            iterator s_end,
                                                            diff_t b,
                                                            Classifier& classifier,
                                                            bool use_equal_buckets);
    
    template <bool kIsParallel>
    bool buildSortingClassifier(iterator begin, iterator end,
                                SharedData* shared);
    
    template <bool kIsParallel>
    diff_t buildPartitioningClassifier(iterator s_begin,
                                    iterator s_end,
                                    diff_t n,
                                    diff_t b,
                                    bool use_equal_buckets,
                                    SharedData* shared);

    void partitionByMerging(iterator begin,
                            iterator end,
                            iterator b_begin,
                            iterator b_end,
                            size_t* bs_begin,
                            bool use_equal_buckets);

    template <bool kEqualBuckets> __attribute__((flatten))
    diff_t classifyLocally(iterator my_begin, iterator my_end);

    inline void parallelClassification(bool use_equal_buckets);

    inline void sequentialClassification(bool use_equal_buckets);

    void moveEmptyBlocks(diff_t my_begin, diff_t my_end,
                         diff_t my_first_empty_block);

    inline int computeOverflowBucket();

    template <bool kEqualBuckets, bool kIsParallel>
    inline int classifyAndReadBlock(int read_bucket);

    template <bool kEqualBuckets, bool kIsParallel>
    inline int swapBlock(diff_t max_off, int dest_bucket,
                         bool current_swap);

    template <bool kEqualBuckets, bool kIsParallel>
    void permuteBlocks();

    inline std::pair<int, diff_t> saveMargins(int last_bucket);

    template<bool post_sort>
    void writeMargins(int first_bucket, int last_bucket,
                      int overflow_bucket, int swap_bucket,
                      diff_t in_swap_buffer);

    template <bool kIsParallel, bool post_sort>
    int partition(iterator begin, iterator end,
                  diff_t* bucket_start, SharedData* shared,
                  int my_id, int num_threads,
                  bool use_equal_buckets);

    inline void processSmallTasks(iterator begin, SharedData& shared);
};

}  // namespace detail

template <class Cfg>
class SequentialSorter;

template <class Cfg>
class ParallelSorter;

template <class It, class Comp>
inline void sort(It begin, It end, Comp comp);

template <class It>
inline void sort(It begin, It end);

#if defined(_REENTRANT) || defined(_OPENMP)
namespace parallel {

template <class It, class Comp>
inline void sort(It begin, It end, Comp comp);

template <class It>
inline void sort(It begin, It end);

}  // namespace parallel
#endif
}  // namespace ips2pa
