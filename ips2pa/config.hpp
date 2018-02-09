/******************************************************************************
 * ips2pa/config.hpp
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
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

#include "thread_pool.hpp"
#include "utils.hpp"

#ifndef IPS2PA_ALLOW_EQUAL_BUCKETS
#define IPS2PA_ALLOW_EQUAL_BUCKETS true
#endif

#ifndef IPS2PA_BASE_CASE_SIZE
#define IPS2PA_BASE_CASE_SIZE 16
#endif

#ifndef IPS2PA_BASE_CASE_MULTIPLIER
#define IPS2PA_BASE_CASE_MULTIPLIER 16
#endif

#ifndef IPS2PA_BLOCK_SIZE
#define IPS2PA_BLOCK_SIZE (2 << 10)
#endif

#ifndef IPS2PA_BUCKET_TYPE
#define IPS2PA_BUCKET_TYPE std::ptrdiff_t
#endif

#ifndef IPS2PA_DATA_ALIGNMENT
#define IPS2PA_DATA_ALIGNMENT (4 << 10)
#endif

#ifndef IPS2PA_EQUAL_BUCKETS_THRESHOLD
#define IPS2PA_EQUAL_BUCKETS_THRESHOLD 5
#endif

#ifndef IPS2PA_LOG_BUCKETS
#define IPS2PA_LOG_BUCKETS 8
#endif

#ifndef IPS2PA_MIN_PARALLEL_BLOCKS_PER_THREAD
#define IPS2PA_MIN_PARALLEL_BLOCKS_PER_THREAD 4
#endif

#ifndef IPS2PA_OVERSAMPLING_FACTOR_PERCENT
#define IPS2PA_OVERSAMPLING_FACTOR_PERCENT 20
#endif

#ifndef IPS2PA_UNROLL_CLASSIFIER
#define IPS2PA_UNROLL_CLASSIFIER 7
#endif

namespace ips2pa {

template <bool AllowEqualBuckets_     = IPS2PA_ALLOW_EQUAL_BUCKETS
        , std::ptrdiff_t BaseCase_    = IPS2PA_BASE_CASE_SIZE
        , std::ptrdiff_t BaseCaseM_   = IPS2PA_BASE_CASE_MULTIPLIER
        , std::ptrdiff_t BlockSize_   = IPS2PA_BLOCK_SIZE
        , class BucketT_              = IPS2PA_BUCKET_TYPE
        , std::size_t DataAlign_      = IPS2PA_DATA_ALIGNMENT
        , std::ptrdiff_t EqualBuckTh_ = IPS2PA_EQUAL_BUCKETS_THRESHOLD
        , int LogBuckets_             = IPS2PA_LOG_BUCKETS
        , std::ptrdiff_t MinParBlks_  = IPS2PA_MIN_PARALLEL_BLOCKS_PER_THREAD
        , int OversampleF_            = IPS2PA_OVERSAMPLING_FACTOR_PERCENT
        , int UnrollClass_            = IPS2PA_UNROLL_CLASSIFIER
        >
struct Config {
    /**
     * The type used for bucket indices in the classifier.
     */
    using bucket_type = BucketT_;

    /**
     * Whether we are on 64 bit or 32 bit.
     */
    static constexpr const bool kIs64Bit = sizeof(std::uintptr_t) == 8;
    static_assert(kIs64Bit || sizeof(std::uintptr_t) == 4, "Architecture must be 32 or 64 bit");

    /**
     * Whether equal buckets can be used.
     */
    static constexpr const bool kAllowEqualBuckets = AllowEqualBuckets_;
    /**
     * Desired base case size.
     */
    static constexpr const std::ptrdiff_t kBaseCaseSize = BaseCase_;
    /**
     * Multiplier for base case threshold.
     */
    static constexpr const int kBaseCaseMultiplier = BaseCaseM_;
    /**
     * Number of bytes in one block.
     */
    static constexpr const std::ptrdiff_t kBlockSizeInBytes = BlockSize_;
    /**
     * Alignment for shared and thread-local data.
     */
    static constexpr const std::size_t kDataAlignment = DataAlign_;
    /**
     * Number of splitters that must be equal before equality buckets are enabled.
     */
    static constexpr const std::ptrdiff_t kEqualBucketsThreshold = EqualBuckTh_;
    /**
     * Logarithm of the maximum number of buckets (excluding equality buckets).
     */
    static constexpr const int kLogBuckets = LogBuckets_;
    /**
     * Minimum number of blocks per thread for which parallelism is used.
     */
    static constexpr const std::ptrdiff_t kMinParallelBlocksPerThread = MinParBlks_;
    static_assert(kMinParallelBlocksPerThread > 0, "Min. blocks per thread must be at least 1.");
    /**
     * How many times the classification loop is unrolled.
     */
    static constexpr const int kUnrollClassifier = UnrollClass_;

    static constexpr const std::ptrdiff_t kSingleLevelThreshold = kBaseCaseSize * (1ul << kLogBuckets);
    static constexpr const std::ptrdiff_t kTwoLevelThreshold = kSingleLevelThreshold * (1ul << kLogBuckets);

    /**
     * The oversampling factor to be used for input of size n.
     */
    static constexpr double oversamplingFactor(std::ptrdiff_t n) {
        return OversampleF_ / 100.0 * detail::log2(n) < 1.0 ? 1.0 : OversampleF_ / 100.0 * detail::log2(n);
    }

    /**
    * Computes the logarithm of the number of buckets to use for input size n.
    */
    static int logBuckets(const std::ptrdiff_t n) {
        if (n <= kSingleLevelThreshold) {
            // Only one more level until we reach the base case, reduce the number of buckets
            return std::max(1ul, detail::log2(n / kBaseCaseSize));
        } else if (n <= kTwoLevelThreshold) {
            // Only two more levels until we reach the base case, split the buckets evenly
            return std::max(1ul, (detail::log2(n / kBaseCaseSize) + 1) / 2);
        } else {
            // Use the maximum number of buckets
            return kLogBuckets;
        }
    }

    /** 
     * Computes the logarithm of the number of buckets used for input size n and bucket size b.
     */
    static int logBucketsPartitioner(const std::ptrdiff_t b) {
        const auto b_log = detail::log2(2 * b - 1);
        if (b_log <= kLogBuckets) {
            return b_log;
        } else if (b_log < kLogBuckets * 2) {
            return (b_log + 1) / 2;
        } else {
            return kLogBuckets;
        }
    }

    /**
     * Returns the number of threads that should be used for the given input range.
     */
    template <class It>
#if defined(_REENTRANT) || defined(_OPENMP)
    static constexpr int numThreadsFor(const It& begin, const It& end, int max_threads) {
        const std::ptrdiff_t blocks = (end - begin) * sizeof(decltype(*begin)) / kBlockSizeInBytes;
        return (blocks < (kMinParallelBlocksPerThread * max_threads)) ? 1 : max_threads;
#else
    static constexpr int numThreadsFor(const It&, const It&, int) {
        return 1;
#endif
    }
};

template <class It_, class Comp_, class Cfg = Config<>
#if defined(_REENTRANT) || defined(_OPENMP)
        , class ThreadPool_ = DefaultThreadPool
#endif
        >
struct ExtendedConfig : public Cfg {
    /**
     * Base config containing user-specified parameters.
     */
    using BaseConfig = Cfg;
    /**
     * The iterator type for the input data.
     */
    using iterator = It_;
    /**
     * The difference type for the iterator.
     */
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    /**
     * The value type of the input data.
     */
    using value_type = typename std::iterator_traits<iterator>::value_type;
    /**
     * The comparison operator.
     */
    using less = Comp_;

#if defined(_REENTRANT) || defined(_OPENMP)
    /**
     * Thread pool for parallel algorithm.
     */
    using ThreadPool = ThreadPool_;

    /**
     * Synchronization support for parallel algorithm.
     */
    using Sync = decltype(std::declval<ThreadPool&>().sync());
#else
    struct Sync {
        constexpr void barrier() const {}
        template <class F>
        constexpr void single(F&&) const {}
    };
#endif

    /**
     * Maximum number of buckets (including equality buckets).
     */
    static constexpr const int kMaxBuckets = 1ul << (Cfg::kLogBuckets + Cfg::kAllowEqualBuckets);

    /**
     * Number of elements in one block.
     */
    static constexpr const difference_type kBlockSize =
            1ul << (detail::log2(
                    Cfg::kBlockSizeInBytes < sizeof(value_type)
                            ? 1
                            : (Cfg::kBlockSizeInBytes / sizeof(value_type))));

    // Redefine applicable constants as difference_type.
    static constexpr const difference_type kBaseCaseSize = Cfg::kBaseCaseSize;
    static constexpr const difference_type kEqualBucketsThreshold = Cfg::kEqualBucketsThreshold;

    // Cannot sort without random access.
    static_assert(std::is_same<typename std::iterator_traits<iterator>::iterator_category,
                  std::random_access_iterator_tag>::value, "Iterator must be a random access iterator.");
    // Number of buckets is limited by switch in classifier
    static_assert(Cfg::kLogBuckets <= 8, "Max. bucket count must be <= 512.");
    // The implementation of the block alignment limits the possible block sizes.
    static_assert((kBlockSize & (kBlockSize - 1)) == 0, "Block size must be a power of two.");
    // The main classifier function assumes that the loop can be unrolled at least once.
    static_assert(Cfg::kUnrollClassifier <= kBaseCaseSize, "Base case size must be larger than unroll factor.");

    /**
     * Aligns an offset to the next block boundary, upwards.
     */
    static constexpr difference_type alignToNextBlock(difference_type p) {
        return (p + kBlockSize - 1) & ~(kBlockSize - 1);
    }
};

#undef IPS2PA_ALLOW_EQUAL_BUCKETS
#undef IPS2PA_BASE_CASE_SIZE
#undef IPS2PA_BASE_CASE_MULTIPLIER
#undef IPS2PA_BLOCK_SIZE
#undef IPS2PA_BUCKET_TYPE
#undef IPS2PA_DATA_ALIGNMENT
#undef IPS2PA_EQUAL_BUCKETS_THRESHOLD
#undef IPS2PA_LOG_BUCKETS
#undef IPS2PA_MIN_PARALLEL_BLOCKS_PER_THREAD
#undef IPS2PA_OVERSAMPLING_FACTOR_PERCENT
#undef IPS2PA_UNROLL_CLASSIFIER

}  // namespace ips2pa
