# In-Place Super Scalar Partitioner (IPS²Pa)

This is the implementation of a k-way multilevel partitioner, an extension of the sorting algorithm presented in the [eponymous paper](https://arxiv.org/abs/1705.02257). 

> We present a partitioning algorithm that works in-place, is
> cache-efficient, avoids branch-mispredictions, and performs work O(k log n) for
> arbitrary inputs. The main algorithmic contribution is a coarse-grained block-based permutation which makes distribution-based algorithms in-place.

## Usage

```C++
#include "ips2pa.hpp"

// sort sequentially
ips2pa::partition(It begin, It end, It s_begin, It s_end, size_t* b_begin, bool use_equal_buckets[, Comp comp, std::ptrdiff_t seed])
```

The parameters `begin` and `end` determine the input data, the parameters `s_begin` and `s_end` describe the splitters, and the parameter `b_begin` points to the output array which stores the bucket sizes. If the parameter `use_equal_buckets` is set to true, the partitioner groups elements together which are equal to splitters.

Make sure to compile with C++11 support. Currently, the code does not compile on Windows.