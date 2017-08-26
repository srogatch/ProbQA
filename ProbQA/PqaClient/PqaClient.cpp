// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

void BenchmarkEvent() {
  const int64_t nIterations = 10 * 1000 * 1000;
  HANDLE hEvent = CreateEvent(nullptr, true, true, nullptr);
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nIterations; i++) {
    WaitForSingleObject(hEvent, INFINITE);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("%.3lf Ops/sec\n", nIterations / nSec);
}

// Taken from: https://www.codeproject.com/Tips/476970/finally-clause-in-Cplusplus
class Finally1 {
  std::function<void(void)> _functor;
public:
  Finally1(const std::function<void(void)> &functor) : _functor(functor) {}
  ~Finally1() {
    _functor();
  }
};

void BenchmarkFunctor() {
  volatile int64_t var = 0;
  const int64_t nIterations = 234567890;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nIterations; i++) {
    Finally1 doFinally([&] {
      var++;
    });
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Functor: %.3lf Ops/sec, var=%lld\n", nIterations / nSec, (long long)var);
}

void BenchmarkObject() {
  volatile int64_t var = 0;
  const int64_t nIterations = 234567890;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nIterations; i++) {
      class Cleaner {
        volatile int64_t* _pVar;
      public:
        Cleaner(volatile int64_t& var) : _pVar(&var) { }
        ~Cleaner() { (*_pVar)++; }
      } c(var);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Object: %.3lf Ops/sec, var=%lld\n", nIterations / nSec, (long long)var);
}

void BenchmarkMSVCpp() {
  volatile int64_t var = 0;
  const int64_t nIterations = 234567890;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nIterations; i++) {
    __try {
    }
    __finally {
      var++;
    }
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("__finally: %.3lf Ops/sec, var=%lld\n", nIterations / nSec, (long long)var);
}

template <typename Func> class Finally4 {
  Func f;
public:
  Finally4(Func&& func) : f(std::forward<Func>(func)) {}
  ~Finally4() { f(); }
  Finally4(const Finally4&) = delete;
  Finally4& operator=(const Finally4&) = delete;
  Finally4(Finally4&&) = delete;
  Finally4& operator=(Finally4&&) = delete;
};

template <typename F> Finally4<F> MakeFinally4(F&& f) {
  return { std::forward<F>(f) };
}

void BenchmarkTemplate() {
  volatile int64_t var = 0;
  const int64_t nIterations = 234567890;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nIterations; i++) {
    auto&& doFinally = MakeFinally4([&] { var++; }); (void)doFinally;
    //C++17: Finally4 doFinally{ [&] { var++; } };
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Template: %.3lf Ops/sec, var=%lld\n", nIterations / nSec, (long long)var);
}

//void BenchmarkMacro() {
//  volatile int64_t var = 0;
//  const int64_t nIterations = 234567890;
//  auto start = std::chrono::high_resolution_clock::now();
//  for (int64_t i = 0; i < nIterations; i++) {
//    SR_FINALLY([&] {
//      var++; 
//    });
//  }
//  auto elapsed = std::chrono::high_resolution_clock::now() - start;
//  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//  printf("Macro: %.3lf Ops/sec, var=%lld\n", nIterations / nSec, (long long)var);
//}

void BenchmarkEmpty() {
  volatile int64_t var = 0;
  const int64_t nIterations = 234567890;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nIterations; i++) {
    var++;
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Empty: %.3lf Ops/sec, var=%lld\n", nIterations / nSec, (long long)var);
}

std::atomic<int64_t> gCounter(0);
const int64_t gnAtomicIterations = 10 * 1000 * 1000;

void CountingThread() {
  for (int64_t i = 0; i < gnAtomicIterations; i++) {
    gCounter.fetch_add(1, std::memory_order_acq_rel);
  }
}

void BenchmarkAtomic() {
  const uint32_t maxThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> thrs;
  thrs.reserve(maxThreads + 1);

  for (uint32_t nThreads = 1; nThreads <= maxThreads; nThreads++) {
    gCounter.store(0, std::memory_order_release);
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs.emplace_back(CountingThread);
    }
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs[i].join();
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("%d threads: %.3lf Ops/sec, counter=%lld\n", (int)nThreads, (nThreads * gnAtomicIterations) / nSec,
      (long long)gCounter.load(std::memory_order_acquire));

    thrs.clear();
  }
}

double *gpdInput;
double *gpdOutput;
const int64_t cnDoubles = 1024 * 1024 * 1024;
const double cDivisor = 3;

void BenchmarkRandFill() {
  std::mt19937_64 rng;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < cnDoubles; i++) {
    gpdInput[i] = rng() / (double(rng())  + 1);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Random fill: %.3lf bytes/sec.\n", cnDoubles * sizeof(double) / nSec);
}

void BenchmarkFastRandFill() {
  SRPlat::SRFastRandom fr;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < cnDoubles; i++) {
    gpdInput[i] = fr.Generate<uint64_t>() / (double(fr.Generate<uint64_t>()) + 1);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Fast random fill: %.3lf bytes/sec.\n", cnDoubles * sizeof(double) / nSec);
}

void BenchmarkCaching() {
  const __m256d divisor = _mm256_set1_pd(cDivisor);
  const __m256d *pSrc = reinterpret_cast<const __m256d*>(gpdInput);
  __m256d *pDest = reinterpret_cast<__m256d*>(gpdOutput);
  int64_t nVects = cnDoubles * sizeof(*gpdInput) / sizeof(*pSrc);
  auto start = std::chrono::high_resolution_clock::now();
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256d dividend = _mm256_load_pd(reinterpret_cast<const double*>(pSrc));
    const __m256d quotient = _mm256_div_pd(dividend, divisor);
    _mm256_store_pd(reinterpret_cast<double*>(pDest), quotient);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Caching: %.3lf bytes/sec.\n", cnDoubles * 2 * sizeof(double) / nSec);
}

void BenchmarkNonCaching() {
  const __m256d divisor = _mm256_set1_pd(cDivisor);
  const __m256d *pSrc = reinterpret_cast<const __m256d*>(gpdInput);
  __m256d *pDest = reinterpret_cast<__m256d*>(gpdOutput);
  int64_t nVects = cnDoubles * sizeof(*gpdInput) / sizeof(*pSrc);
  auto start = std::chrono::high_resolution_clock::now();
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256d dividend = _mm256_castsi256_pd(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(pSrc)));
    const __m256d quotient = _mm256_div_pd(dividend, divisor);
    _mm256_stream_pd(reinterpret_cast<double*>(pDest), quotient);
  }
  _mm_sfence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Non-caching: %.3lf bytes/sec.\n", cnDoubles * 2 * sizeof(double) / nSec);
}

void BenchmarkMemcpy() {
  auto start = std::chrono::high_resolution_clock::now();
  memcpy(gpdOutput, gpdInput, sizeof(double)*SRPlat::SRCast::ToSizeT(cnDoubles));
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("memcpy(): %.3lf bytes/sec.\n", cnDoubles * 2 * sizeof(double) / nSec);
}

void BenchmarkStreamCopy() {
  auto start = std::chrono::high_resolution_clock::now();
  const __m256i *pSrc = reinterpret_cast<const __m256i*>(gpdInput);
  __m256i *pDest = reinterpret_cast<__m256i*>(gpdOutput);
  int64_t nVects = cnDoubles * sizeof(*gpdInput) / sizeof(*pSrc);
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256i loaded = _mm256_stream_load_si256(pSrc);
    _mm256_stream_si256(pDest, loaded);
  }
  _mm_sfence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Stream copy: %.3lf bytes/sec.\n", cnDoubles * 2 * sizeof(double) / nSec);
}

void AsyncStreamCopy(__m256i *pDest, const __m256i *pSrc, int64_t nVects) {
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256i loaded = _mm256_stream_load_si256(pSrc);
    _mm256_stream_si256(pDest, loaded);
  }
}

void BenchmarkMultithreadStreamCopy() {
  const uint32_t maxThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> thrs;
  thrs.reserve(maxThreads + 1);

  const __m256i *pSrc = reinterpret_cast<const __m256i*>(gpdInput);
  __m256i *pDest = reinterpret_cast<__m256i*>(gpdOutput);
  const int64_t nVects = cnDoubles * sizeof(*gpdInput) / sizeof(*pSrc);

  for (uint32_t nThreads = 1; nThreads <= maxThreads; nThreads++) {
    auto start = std::chrono::high_resolution_clock::now();
    lldiv_t perWorker = div((long long)nVects, (long long)nThreads);
    int64_t nextStart = 0;
    for (uint32_t i = 0; i < nThreads; i++) {
      const int64_t curStart = nextStart;
      nextStart += perWorker.quot;
      if ((long long)i < perWorker.rem) {
        nextStart++;
      }
      thrs.emplace_back(AsyncStreamCopy, pDest + curStart, pSrc+curStart, nextStart-curStart);
    }
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs[i].join();
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("Stream copy %d threads: %.3lf bytes/sec\n", (int)nThreads, cnDoubles * 2 * sizeof(double) / nSec);

    thrs.clear();
  }
}

void MultibenchMemory() {
  gpdInput = (double*)_mm_malloc(SRPlat::SRCast::ToSizeT(cnDoubles) * sizeof(double), sizeof(__m256d));
  gpdOutput = (double*)_mm_malloc(SRPlat::SRCast::ToSizeT(cnDoubles) * sizeof(double), sizeof(__m256d));

  BenchmarkRandFill();

  BenchmarkFastRandFill();
  double s = 0;

  BenchmarkCaching();

  for (int64_t i = 0; i < cnDoubles; i++) {
    s += gpdOutput[i];
  }

  BenchmarkNonCaching();

  for (int64_t i = 0; i < cnDoubles; i++) {
    s -= gpdOutput[i];
  }
  printf("Control sum: %lf\n", s);

  BenchmarkMemcpy();

  for (int64_t i = 0; i < cnDoubles; i++) {
    s += gpdOutput[i];
  }

  BenchmarkStreamCopy();

  for (int64_t i = 0; i < cnDoubles; i++) {
    s -= gpdOutput[i];
  }
  printf("Control sum: %lf\n", s);

  BenchmarkMultithreadStreamCopy();
  for (int64_t i = 0; i < cnDoubles; i++) {
    s += gpdOutput[i];
  }
  printf("Control sum: %lf\n", s);

  _mm_free(gpdInput);
  _mm_free(gpdOutput);
}

void BenchmarkStdFunctionMemPool() {
  const int64_t nItems = 32 * 1024 * 1024;
  typedef SRPlat::SRMemPool<SRPlat::SRSimd::_cLogNBits, 1 << 10> TMemPool;
  typedef SRPlat::SRMPAllocator<char, TMemPool> TAllocator;
  TMemPool memPool;
  TAllocator alloc(memPool);

  std::queue<std::function<void()>, std::deque<std::function<void()>, TAllocator>> qu(alloc);
  volatile int64_t sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nItems; i++) {
    qu.emplace(std::allocator_arg, alloc, [&sum]() { sum++; });
  }

  while (!qu.empty()) {
    std::function<void()> f(std::move(qu.front()));
    qu.pop();
    f();
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("std::function with MemPool: %.3lf push&pop per second. Sum: %lld\n", nItems / nSec,
    (long long)sum);
}

void BenchmarkStdFunctionStdAlloc() {
  const int64_t nItems = 32 * 1024 * 1024;

  std::queue<std::function<void()>> qu;
  volatile int64_t sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nItems; i++) {
    qu.emplace([&sum]() { sum++; });
  }

  while (!qu.empty()) {
    std::function<void()> f(std::move(qu.front()));
    qu.pop();
    f();
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("std::function with std alloc: %.3lf push&pop per second. Sum: %lld\n", nItems / nSec,
    (long long)sum);
}

class BaseSubtask {
public:
  virtual ~BaseSubtask() { }
  virtual void Run() = 0;
};

class TestSubtask : public BaseSubtask {
  int64_t *_pSum;
public:
  explicit TestSubtask(int64_t& sum) : _pSum(&sum) { }
  virtual void Run() override final {
    (*_pSum)++;
  }
};

void BenchmarkSubtask() {
  const int64_t nItems = 32 * 1024 * 1024;

  std::queue<BaseSubtask*> qu;
  int64_t sum = 0;
  TestSubtask tst(sum);
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nItems; i++) {
    qu.emplace(&tst);
  }

  while (!qu.empty()) {
    BaseSubtask *pBst = qu.front();
    qu.pop();
    pBst->Run();
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Subtask: %.3lf push&pop per second. Sum: %lld\n", nItems / nSec,
    (long long)sum);
}

// This one is 55 cycles per push&pop&call , 71 549 818 triples per second.
void BenchmarkSmallQueue() {
  const int64_t nItems = 32 * 1024 * 1024;

  std::queue<BaseSubtask*> qu;
  int64_t sum = 0;
  TestSubtask tst(sum);
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nItems; i++) {
    qu.emplace(&tst);
    if (qu.size() >= 1024) {
      do {
        BaseSubtask *pBst = qu.front();
        qu.pop();
        pBst->Run();
      } while (!qu.empty());
    }
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Small queue: %.3lf push&pop per second. Sum: %lld\n", nItems / nSec,
    (long long)sum);
}

template <typename Func> class LambdaSubtask : public BaseSubtask {
  Func _f;
public:
  LambdaSubtask(Func&& func) : _f(std::forward<Func>(func)) {}
  LambdaSubtask(LambdaSubtask&) = delete;
  LambdaSubtask& operator=(const LambdaSubtask&) = delete;
  LambdaSubtask(LambdaSubtask&&) = delete;
  LambdaSubtask& operator=(LambdaSubtask&&) = delete;

  virtual void Run() override final {
    _f();
  }
};

template <typename F> LambdaSubtask<F> MakeLambdaSubtask(F&& f) {
  return { std::forward<F>(f) };
}

void BenchmarkLambda() {
  const int64_t nItems = 32 * 1024 * 1024;

  std::queue<BaseSubtask*> qu;
  int64_t sum = 0;
  auto fnOp = [&sum]() { sum++; };
  LambdaSubtask<decltype(fnOp)> lst(std::move(fnOp));
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nItems; i++) {
    qu.emplace(&lst);
    if (qu.size() >= 1024) {
      do {
        BaseSubtask *pBst = qu.front();
        qu.pop();
        pBst->Run();
      } while (!qu.empty());
    }
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Small queue: %.3lf push&pop per second. Sum: %lld\n", nItems / nSec,
    (long long)sum);
}

typedef uint32_t TCacheLineEntry;
const int64_t cnCacheEntryIncrements = 100 * 1000 * 1000;
const uint16_t gCacheLineBytes = 64;
volatile TCacheLineEntry *gpCacheLine = nullptr;

void CacheEntryIncrement(const uint32_t iWorker) {
  for (int64_t i = 0; i < cnCacheEntryIncrements; i++) {
    gpCacheLine[iWorker]++;
  }
}

void BenchmarkCacheLine() {
  const uint32_t maxThreads = std::thread::hardware_concurrency();
  gpCacheLine = reinterpret_cast<TCacheLineEntry*>(_mm_malloc(maxThreads * sizeof(TCacheLineEntry), gCacheLineBytes));
  std::vector<std::thread> thrs;
  thrs.reserve(maxThreads + 1);

  for (uint32_t nThreads = 1; nThreads <= maxThreads; nThreads++) {
    std::memset(const_cast<TCacheLineEntry*>(gpCacheLine), 0, maxThreads * sizeof(TCacheLineEntry));
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs.emplace_back(CacheEntryIncrement, i);
    }
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs[i].join();
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    int64_t sum = 0;
    for (uint32_t i = 0; i < nThreads; i++) {
      sum += gpCacheLine[i];
    }
    printf("Contended cache line: %d threads give %.3lf Ops/sec, sum=%lld\n", (int)nThreads,
      (nThreads * cnCacheEntryIncrements) / nSec, (long long)sum);

    thrs.clear();
  }
  _mm_free(const_cast<TCacheLineEntry*>(gpCacheLine));
}

const int64_t cLogsStart = 1000LL * 1000 * 1000;
const int64_t cnLogs = 1000LL * 1000 * 1000;

void BenchmarkLog2() {
  double sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for(int64_t i=1; i<=cnLogs; i++) {
    const double x = double(i + cLogsStart);
    sum += std::log2(x);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("std::log2: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void BenchmarkFpuLog2() {
  double sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i++) {
    const double x = double(i + cLogsStart);
    sum += SRPlat::SRLog2MulD(x, 1);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("FPU Log2: %.6lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void BenchmarkLn() {
  double sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i++) {
    const double x = double(i + cLogsStart);
    sum += std::log(x);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Ln: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum / std::log(2.0));
}

void BenchmarkLog2Quads() {
  double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i+=4) {
    sum0 += std::log2(double(i));
    sum1 += std::log2(double(i+1));
    sum2 += std::log2(double(i + 2));
    sum3 += std::log2(double(i + 3));
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  double sum = sum0 + sum1 + sum2 + sum3;
  printf("Quad Log2: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

__m256d __vectorcall Log2inl(__m256d x) {
  const __m256i cDoubleExpMask = _mm256_set1_epi64x(0x7ffULL << 52);
  const __m256i exps64 = _mm256_srli_epi64(_mm256_and_si256(cDoubleExpMask, _mm256_castpd_si256(x)), 52);
  const __m256i cTo32bitExp = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
  const __m256i exps32_avx = _mm256_permutevar8x32_epi32(exps64, cTo32bitExp);
  const __m128i exps32_sse = _mm256_castsi256_si128(exps32_avx);
  const __m128i cExpNormalizer = _mm_set1_epi32(1023);
  const __m128i normExps = _mm_sub_epi32(exps32_sse, cExpNormalizer);
  const __m256d expsPD = _mm256_cvtepi32_pd(normExps);
  const __m256i cDoubleExp0 = _mm256_set1_epi64x(1023ULL << 52);
  const __m256d y = _mm256_or_pd(_mm256_castsi256_pd(cDoubleExp0),
    _mm256_andnot_pd(_mm256_castsi256_pd(cDoubleExpMask), x));

  // Calculate t=(y-1)/(y+1) and t**2
  const __m256d cVect1 = _mm256_set1_pd(1.0);
  const __m256d tNum = _mm256_sub_pd(y, cVect1);
  const __m256d tDen = _mm256_add_pd(y, cVect1);
  const __m256d t = _mm256_div_pd(tNum, tDen);
  const __m256d t2 = _mm256_mul_pd(t, t); // t**2

  const __m256d t3 = _mm256_mul_pd(t, t2); // t**3
  const __m256d terms01 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 3), t3, t);
  const __m256d t5 = _mm256_mul_pd(t3, t2); // t**5
  const __m256d terms012 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 5), t5, terms01);
  const __m256d t7 = _mm256_mul_pd(t5, t2); // t**7
  const __m256d terms0123 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 7), t7, terms012);
  const __m256d t9 = _mm256_mul_pd(t7, t2); // t**9
  const __m256d terms01234 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 9), t9, terms0123);
  const __m256d t11 = _mm256_mul_pd(t9, t2); // t**9
  const __m256d terms012345 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 11), t11, terms01234);

  const double cCommMul = 2.0 / 0.693147180559945309417; // 2.0/ln(2)
  const __m256d log2_y = _mm256_mul_pd(terms012345, _mm256_set1_pd(cCommMul));
  const __m256d log2_x = _mm256_add_pd(log2_y, expsPD);

  return log2_x;
}

void BenchmarkLog2VectInl() {
  __m256d sums = _mm256_setzero_pd();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const __m256d x = _mm256_set_pd(double(i + 3), double(i + 2), double(i + 1), double(i));
    const __m256d logs = Log2inl(x);
    sums = _mm256_add_pd(sums, logs);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  double sum = sums.m256d_f64[0] + sums.m256d_f64[1] + sums.m256d_f64[2] + sums.m256d_f64[3];
  printf("Vect Log2inl: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

// 5 terms Vect Log2 : 405268490.375 Ops / sec calculated 2513272973.836
// 6 terms Vect Log2: 352875583.127 Ops/sec calculated 2513272985.407

namespace {
  const __m256i gDoubleNotExp = _mm256_set1_epi64x(~(0x7ffULL << 52));
  const __m256d gDoubleExp0 = _mm256_castsi256_pd(_mm256_set1_epi64x(1023ULL << 52));
  //const __m256i gDoubleExpM1 = _mm256_set1_epi64x(1022ULL << 52); // exponent "-1"
  const __m256d gLowestExpBits = _mm256_castsi256_pd(_mm256_set1_epi64x(1ULL << 52));

  const __m128i gExpNorm0 = _mm_set1_epi32(1023);
  const __m128i gExpNormM1 = _mm_set1_epi32(1022);

  const __m256i gHigh32Permute = _mm256_set_epi32(0, 0, 0, 0, 7, 5, 3, 1);

  const __m256d gCommMul1 = _mm256_set1_pd(2.0 / 0.693147180559945309417); // 2.0/ln(2)
  const __m256d gSqrt2 = _mm256_set1_pd(1.4142135623730950488016887242097); // sqrt(2)
  const __m256d gCoeff1 = _mm256_set1_pd(1.0 / 3);
  const __m256d gCoeff2 = _mm256_set1_pd(1.0 / 5);
  const __m256d gCoeff3 = _mm256_set1_pd(1.0 / 7);
  const __m256d gCoeff4 = _mm256_set1_pd(1.0 / 9);
  const __m256d gCoeff5 = _mm256_set1_pd(1.0 / 11);
  const __m256d gVect1 = _mm256_set1_pd(1.0);

  const __m256d gCommMulSqrt = _mm256_set1_pd(4.0 / 0.693147180559945309417); // 4.0/ln(2)
}

// Using blendv: 332348915.845 Ops / sec calculated 28454657829.372330
// Using logical: 332701754.503 Ops/sec calculated 28454657829.372330
// std::log2(): 95186467.911 Ops/sec calculated 28454657829.372597
__m256d __vectorcall Log2(__m256d x) {
  const __m256d yClearExp = _mm256_and_pd(_mm256_castsi256_pd(gDoubleNotExp), x);
  const __m256d yExp0 = _mm256_or_pd(yClearExp, gDoubleExp0);
  //const __m256d yExpM1 = _mm256_or_pd(yClearExp, _mm256_castsi256_pd(gDoubleExpM1));
  const __m256i cmpResAvx = _mm256_castpd_si256(_mm256_cmp_pd(yExp0, gSqrt2, _CMP_GT_OQ));
  //const __m256d y = _mm256_blendv_pd(yExp0, yExpM1, _mm256_castsi256_pd(cmpResAvx));
  const __m256d y = _mm256_xor_pd(yExp0, _mm256_and_pd(_mm256_castsi256_pd(cmpResAvx), gLowestExpBits));
  const __m128i cmpResSse = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(cmpResAvx, gHigh32Permute));

  // Calculate t=(y-1)/(y+1) and t**2
  const __m256d tNum = _mm256_sub_pd(y, gVect1);
  const __m256d tDen = _mm256_add_pd(y, gVect1);
  const __m256d t = _mm256_div_pd(tNum, tDen);
  const __m256d t2 = _mm256_mul_pd(t, t); // t**2

  const __m256d t3 = _mm256_mul_pd(t, t2); // t**3
  const __m256d terms01 = _mm256_fmadd_pd(gCoeff1, t3, t);
  const __m256d t5 = _mm256_mul_pd(t3, t2); // t**5
  const __m256d terms012 = _mm256_fmadd_pd(gCoeff2, t5, terms01);
  const __m256d t7 = _mm256_mul_pd(t5, t2); // t**7
  const __m256d terms0123 = _mm256_fmadd_pd(gCoeff3, t7, terms012);
  const __m256d t9 = _mm256_mul_pd(t7, t2); // t**9
  const __m256d terms01234 = _mm256_fmadd_pd(gCoeff4, t9, terms0123);
  const __m256d t11 = _mm256_mul_pd(t9, t2); // t**11
  const __m256d terms012345 = _mm256_fmadd_pd(gCoeff5, t11, terms01234);

  const __m128i high32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), gHigh32Permute));
  // This requires that x is non-negative, because the sign bit is not cleared before computing the exponent.
  const __m128i exps32 = _mm_srai_epi32(high32, 20);
  const __m128i normalizer = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(gExpNorm0), _mm_castsi128_ps(gExpNormM1),
    _mm_castsi128_ps(cmpResSse)));
  const __m128i normExps = _mm_sub_epi32(exps32, normalizer);
  const __m256d expsPD = _mm256_cvtepi32_pd(normExps);

  const __m256d log2_x = _mm256_fmadd_pd(terms012345, gCommMul1, expsPD);
  return log2_x;
}

namespace {
  // The limit is 19 because we process only high 32 bits of doubles, and out of 20 bits of mantissa there, 1 bit is
  //   used for rounding.
  const uint8_t cnLog2TblBits = 10; // 1024 numbers times 8 bytes = 8KB.
  const uint16_t cZeroExp = 1023;
  const __m128i cSseMantTblMask = _mm_set1_epi32((1 << cnLog2TblBits) - 1);
  const __m128i cSseRoundingMantTblMask = _mm_set1_epi32( ((1 << (cnLog2TblBits+1)) - 1) << (20-cnLog2TblBits-1) );
  const __m128i cSseRoundingBit = _mm_set1_epi32(1 <<(20-cnLog2TblBits-1));
  const __m256i cAvxExp2YMask = _mm256_set1_epi64x( ~((1ULL << (52-cnLog2TblBits)) - 1) );
  const __m256d cPlusBit = _mm256_castsi256_pd(_mm256_set1_epi64x(1ULL << (52 - cnLog2TblBits - 1)));
  double gLog2Table[(1 << cnLog2TblBits)+1];
  double gPlusLog2Table[1 << cnLog2TblBits]; // plus |cnLog2TblBits|th highest bit
}

void InitLog2Table() {
  for(uint32_t i=0; i<(1<<cnLog2TblBits); i++) {
    const uint64_t iZ = (uint64_t(cZeroExp) << 52) | (uint64_t(i) << (52 - cnLog2TblBits));
    const double z = *reinterpret_cast<const double*>(&iZ);
    const double l2z = std::log2(z);
    gLog2Table[i] = l2z;

    const uint64_t iZp = iZ | (1ULL << (52 - cnLog2TblBits - 1));
    const double zp = *reinterpret_cast<const double*>(&iZp);
    const double l2zp = std::log2(zp);
    gPlusLog2Table[i] = l2zp;
  }
  gLog2Table[1 << cnLog2TblBits] = 1;
}

// With one term and 10 bit table, this gives relative error 0.39684 * 10**-13
__m256d __vectorcall Log2tbl(__m256d x) {
  const __m256d zClearExp = _mm256_and_pd(_mm256_castsi256_pd(gDoubleNotExp), x);
  const __m256d z = _mm256_or_pd(zClearExp, gDoubleExp0);

  // Permuting floats seems slower
  //const __m128i high32 = _mm_castps_si128( _mm256_castps256_ps128(
  //  _mm256_permutevar8x32_ps(_mm256_castpd_ps(x), gHigh32Permute) ));
  const __m128i high32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), gHigh32Permute));
  // This requires that x is non-negative, because the sign bit is not cleared before computing the exponent.
  const __m128i exps32 = _mm_srai_epi32(high32, 20);
  const __m128i normExps = _mm_sub_epi32(exps32, gExpNorm0);
  // if |leading| is computed here, the performance is 482953667.357 Ops/sec
  
  // Compute y as approximately equal to log2(z)
  const __m128i indexes = _mm_and_si128(cSseMantTblMask, _mm_srai_epi32(high32, 20 - cnLog2TblBits));
  const __m256d y = _mm256_i32gather_pd(gLog2Table, indexes, /*number of bytes per item*/ 8);
  // Compute A as z/exp2(y)
  const __m256d exp2_Y = _mm256_and_pd(z, _mm256_castsi256_pd(cAvxExp2YMask));
  //const __m256d A = _mm256_div_pd(z, exp2_Y);

  // Calculate t=(A-1)/(A+1)
  //const __m256d tNum = _mm256_sub_pd(A, gVect1);
  //const __m256d tNum = _mm256_div_pd(_mm256_sub_pd(z, exp2_Y), exp2_Y);
  const __m256d tNum = _mm256_sub_pd(z, exp2_Y);
  //const __m256d tDen = _mm256_add_pd(A, gVect1);
  //const __m256d tDen = _mm256_div_pd(_mm256_add_pd(z, exp2_Y), exp2_Y);
  const __m256d tDen = _mm256_add_pd(z, exp2_Y); // both numerator and denominator would be divided by exp2_Y

  const __m256d t = _mm256_div_pd(tNum, tDen);
  //const __m256d t2 = _mm256_mul_pd(t, t); // t**2

  //const __m256d t3 = _mm256_mul_pd(t, t2); // t**3
  //const __m256d terms01 = _mm256_fmadd_pd(gCoeff1, t3, t);
  //const __m256d t5 = _mm256_mul_pd(t3, t2); // t**5
  //const __m256d terms012 = _mm256_fmadd_pd(gCoeff2, t5, terms01);
  //const __m256d t7 = _mm256_mul_pd(t5, t2); // t**7
  //const __m256d terms0123 = _mm256_fmadd_pd(gCoeff3, t7, terms012);
  //const __m256d t9 = _mm256_mul_pd(t7, t2); // t**9
  //const __m256d terms01234 = _mm256_fmadd_pd(gCoeff4, t9, terms0123);
  //const __m256d t11 = _mm256_mul_pd(t9, t2); // t**11
  //const __m256d terms012345 = _mm256_fmadd_pd(gCoeff5, t11, terms01234);

  const __m256d log2_z = _mm256_fmadd_pd(/*terms012345*/ t, gCommMul1, y);

  const __m256d leading = _mm256_cvtepi32_pd(normExps); // leading integer part for the logarithm

  const __m256d log2_x = _mm256_add_pd(log2_z, leading);
  return log2_x;
}

__m256d __vectorcall Log2tblPlus(__m256d x) {
  const __m256d zClearExp = _mm256_and_pd(_mm256_castsi256_pd(gDoubleNotExp), x);
  const __m256d z = _mm256_or_pd(zClearExp, gDoubleExp0);

  const __m128i high32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), gHigh32Permute));
  // This requires that x is non-negative, because the sign bit is not cleared before computing the exponent.
  const __m128i exps32 = _mm_srai_epi32(high32, 20);
  const __m128i normExps = _mm_sub_epi32(exps32, gExpNorm0);

  // Compute y as approximately equal to log2(z)
  const __m128i indexes = _mm_and_si128(cSseMantTblMask, _mm_srai_epi32(high32, 20 - cnLog2TblBits));
  //const __m256d y = _mm256_i32gather_pd(gPlusLog2Table, indexes, /*number of bytes per item*/ 8);
  const __m256d y = _mm256_set_pd(gPlusLog2Table[indexes.m128i_u32[3]], gPlusLog2Table[indexes.m128i_u32[2]],
    gPlusLog2Table[indexes.m128i_u32[1]], gPlusLog2Table[indexes.m128i_u32[0]]);
  // Compute A as z/exp2(y)
  const __m256d exp2_Y = _mm256_or_pd(cPlusBit, _mm256_and_pd(z, _mm256_castsi256_pd(cAvxExp2YMask)));
  //const __m256d A = _mm256_div_pd(z, exp2_Y);

  // Calculate t=(A-1)/(A+1)
  //const __m256d tNum = _mm256_sub_pd(A, gVect1);
  //const __m256d tNum = _mm256_div_pd(_mm256_sub_pd(z, exp2_Y), exp2_Y);
  const __m256d tNum = _mm256_sub_pd(z, exp2_Y);
  //const __m256d tDen = _mm256_add_pd(A, gVect1);
  //const __m256d tDen = _mm256_div_pd(_mm256_add_pd(z, exp2_Y), exp2_Y);
  const __m256d tDen = _mm256_add_pd(z, exp2_Y); // both numerator and denominator would be divided by exp2_Y

  const __m256d t = _mm256_div_pd(tNum, tDen);

  const __m256d log2_z = _mm256_fmadd_pd(/*terms012345*/ t, gCommMul1, y);

  const __m256d leading = _mm256_cvtepi32_pd(normExps); // leading integer part for the logarithm

  const __m256d log2_x = _mm256_add_pd(log2_z, leading);
  return log2_x;
}

__m256d __vectorcall Log2tblPrec(__m256d x) {
  const __m256d zClearExp = _mm256_and_pd(_mm256_castsi256_pd(gDoubleNotExp), x);
  const __m256d z = _mm256_or_pd(zClearExp, gDoubleExp0);

  const __m128i high32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), gHigh32Permute));
  // This requires that x is non-negative, because the sign bit is not cleared before computing the exponent.
  const __m128i exps32 = _mm_srai_epi32(high32, 20);
  const __m128i normExps = _mm_sub_epi32(exps32, gExpNorm0);
  const __m256d leading = _mm256_cvtepi32_pd(normExps); // leading integer part for the logarithm

  // Compute y as approximately equal to log2(z)

  // Depending on |cnLog2TblBits|th highest bit of mantissa (with the highest bit at index 0), it may be better to
  //   increment the index.
  const __m128i indexes = _mm_srai_epi32(
    _mm_add_epi32(_mm_and_si128(cSseRoundingMantTblMask, high32), cSseRoundingBit), 20 - cnLog2TblBits);
  const __m256d y = _mm256_i32gather_pd(gLog2Table, indexes, /*number of bytes per item*/ 8);
  // Compute A as z/exp2(y)
  //const __m256d exp2_Y = _mm256_and_pd(z, _mm256_castsi256_pd(cAvxExp2YMask));
  const __m256d exp2_Y = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(gDoubleExp0),
    _mm256_slli_epi64(_mm256_cvtepu32_epi64(indexes), 52-cnLog2TblBits) ) );
  // const __m256d A = _mm256_div_pd(z, exp2_Y);

  // Calculate t=(A-1)/(A+1)
  //const __m256d tNum = _mm256_div_pd(_mm256_sub_pd(z, exp2_Y), exp2_Y);
  const __m256d tNum = _mm256_sub_pd(z, exp2_Y);
  //const __m256d tDen = _mm256_div_pd(_mm256_add_pd(z, exp2_Y), exp2_Y);
  const __m256d tDen = _mm256_add_pd(z, exp2_Y);
  const __m256d t = _mm256_div_pd(tNum, tDen); // both numerator and denominator would be divided by exp2_Y

  const __m256d log2_z = _mm256_fmadd_pd(t, gCommMul1, y);
  const __m256d log2_x = _mm256_add_pd(log2_z, leading);
  return log2_x;
}

__m256d __vectorcall Log2sqrt(__m256d x) {
  const __m256d y = _mm256_sqrt_pd(_mm256_or_pd(gDoubleExp0,
    _mm256_and_pd(_mm256_castsi256_pd(gDoubleNotExp), x)));

  // Calculate t=(y-1)/(y+1) and t**2
  const __m256d tNum = _mm256_sub_pd(y, gVect1);
  const __m256d tDen = _mm256_add_pd(y, gVect1);
  const __m256d t = _mm256_div_pd(tNum, tDen);
  const __m256d t2 = _mm256_mul_pd(t, t); // t**2

  const __m256d t3 = _mm256_mul_pd(t, t2); // t**3
  const __m256d terms01 = _mm256_fmadd_pd(gCoeff1, t3, t);
  const __m256d t5 = _mm256_mul_pd(t3, t2); // t**5
  const __m256d terms012 = _mm256_fmadd_pd(gCoeff2, t5, terms01);
  const __m256d t7 = _mm256_mul_pd(t5, t2); // t**7
  const __m256d terms0123 = _mm256_fmadd_pd(gCoeff3, t7, terms012);
  const __m256d t9 = _mm256_mul_pd(t7, t2); // t**9
  const __m256d terms01234 = _mm256_fmadd_pd(gCoeff4, t9, terms0123);
  const __m256d t11 = _mm256_mul_pd(t9, t2); // t**9
  const __m256d terms012345 = _mm256_fmadd_pd(gCoeff5, t11, terms01234);

  const __m128i high32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), gHigh32Permute));
  // This requires that x is non-negative, because the sign bit is not cleared before computing the exponent.
  const __m128i exps32 = _mm_srai_epi32(high32, 20);
  const __m128i normExps = _mm_sub_epi32(exps32, gExpNorm0);
  const __m256d expsPD = _mm256_cvtepi32_pd(normExps);

  const __m256d log2_x = _mm256_fmadd_pd(terms012345, gCommMulSqrt, expsPD);
  return log2_x;
}

void BenchmarkLog2tblVect() {
  __m256d sums = _mm256_setzero_pd();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const __m256d x = _mm256_set_pd(double(i + cLogsStart + 3), double(i + cLogsStart + 2), double(i + cLogsStart + 1),
      double(i + cLogsStart));
    const __m256d logs = Log2tbl(x);
    sums = _mm256_add_pd(sums, logs);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  double sum = sums.m256d_f64[0] + sums.m256d_f64[1] + sums.m256d_f64[2] + sums.m256d_f64[3];
  printf("Vect Log2tbl: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void BenchmarkLog2tblPlus() {
  __m256d sums = _mm256_setzero_pd();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const __m256d x = _mm256_set_pd(double(i + cLogsStart + 3), double(i + cLogsStart + 2), double(i + cLogsStart + 1),
      double(i + cLogsStart));
    const __m256d logs = Log2tblPlus(x);
    sums = _mm256_add_pd(sums, logs);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  double sum = sums.m256d_f64[0] + sums.m256d_f64[1] + sums.m256d_f64[2] + sums.m256d_f64[3];
  printf("Vect Log2tblPlus: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void BenchmarkLog2tblPrec() {
  __m256d sums = _mm256_setzero_pd();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const __m256d x = _mm256_set_pd(double(i + cLogsStart + 3), double(i + cLogsStart + 2), double(i + cLogsStart + 1),
      double(i + cLogsStart));
    const __m256d logs = Log2tblPrec(x);
    sums = _mm256_add_pd(sums, logs);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  double sum = sums.m256d_f64[0] + sums.m256d_f64[1] + sums.m256d_f64[2] + sums.m256d_f64[3];
  printf("Vect Log2tblPrec: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void BenchmarkLog2Vect() {
  __m256d sums = _mm256_setzero_pd();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const __m256d x = _mm256_set_pd(double(i + cLogsStart +3), double(i + cLogsStart +2), double(i + cLogsStart +1),
      double(i + cLogsStart));
    const __m256d logs = Log2(x);
    sums = _mm256_add_pd(sums, logs);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  double sum = sums.m256d_f64[0] + sums.m256d_f64[1] + sums.m256d_f64[2] + sums.m256d_f64[3];
  printf("Vect Log2: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void AsyncLog2Vect(__m256d *pSums) {
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const __m256d x = _mm256_set_pd(double(i + 3), double(i + 2), double(i + 1), double(i));
    const __m256d logs = Log2(x);
    *pSums = _mm256_add_pd(*pSums, logs);
  }
}

void BenchmarkLog2VectThreads() {
  __m256d sums[32];
  for(uint32_t i=1; i<=std::thread::hardware_concurrency(); i++) {
    std::vector<std::thread> thrs;
    thrs.reserve(i);
    auto start = std::chrono::high_resolution_clock::now();
    for(uint32_t j=0; j<i; j++) {
      sums[2*j] = _mm256_setzero_pd();
      thrs.emplace_back(AsyncLog2Vect, sums + 2*j);
    }
    for(uint32_t j=0; j<i; j++) {
      thrs[j].join();
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    double sum = sums[0].m256d_f64[0] + sums[0].m256d_f64[1] + sums[0].m256d_f64[2] + sums[0].m256d_f64[3];
    printf("%d threads VectLog2: %.3lf Ops/sec calculated %.6lf\n", (int)i, cnLogs / nSec, sum);
  }
}

void AsyncLn(double *pSum) {
  for (int64_t i = 1; i <= cnLogs; i++) {
    *pSum += std::log(double(i));
  }
}

void BenchmarkLnThreads() {
  double sums[128];
  for (uint32_t i = 1; i <= std::thread::hardware_concurrency(); i++) {
    std::vector<std::thread> thrs;
    thrs.reserve(i);
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t j = 0; j<i; j++) {
      sums[8 * j] = 0;
      thrs.emplace_back(AsyncLn, sums + 8 * j);
    }
    for (uint32_t j = 0; j<i; j++) {
      thrs[j].join();
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("%d threads Ln: %.3lf Ops/sec calculated %.6lf\n", int(i), cnLogs / nSec, sums[0]);
  }
}

int __cdecl main() {
  //std::atomic<double> test1;
  //bool test2 = test1.is_lock_free();
  //std::cout << test2 << std::endl;

  //BenchmarkFunctor();
  //BenchmarkObject();
  //BenchmarkMSVCpp();
  //BenchmarkTemplate();
  //BenchmarkMacro();
  //BenchmarkEmpty();
  //BenchmarkAtomic();
  //MultibenchMemory();

  //BenchmarkStdFunctionMemPool();
  //BenchmarkStdFunctionStdAlloc();
  //BenchmarkSubtask();
  //BenchmarkSmallQueue();
  //BenchmarkLambda();

  //BenchmarkCacheLine();
  //BenchmarkLog2Quads();
  //BenchmarkLog2VectInl();
  //BenchmarkLnThreads();
  //BenchmarkLog2VectThreads();

  InitLog2Table();
  BenchmarkLog2tblPlus();
  BenchmarkLog2tblPrec();
  BenchmarkLog2tblVect();
  BenchmarkLog2Vect();
  BenchmarkLn();
  BenchmarkLog2();
  BenchmarkFpuLog2();
  return 0;
}
