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
    gpdInput[i] = rng() / (double(rng()) + 1);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Random fill: %.3lf bytes/sec.\n", cnDoubles * sizeof(double) / nSec);
}

void BenchmarkFastRandFill() {
  SRPlat::SRFastRandom fr;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < cnDoubles; i++) {
    gpdInput[i] = fr.Generate64() / (double(fr.Generate64()) + 1);
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
      thrs.emplace_back(AsyncStreamCopy, pDest + curStart, pSrc + curStart, nextStart - curStart);
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

//void BenchmarkStdFunctionMemPool() {
//  const int64_t nItems = 32 * 1024 * 1024;
//  typedef SRPlat::SRMemPool<SRPlat::SRSimd::_cLogNBits, 1 << 10> TMemPool;
//  typedef SRPlat::SRMPAllocator<std::function<void()>, TMemPool> TAllocator;
//  TMemPool memPool;
//  TAllocator alloc(memPool);
//
//  std::queue<std::function<void()>, std::deque<std::function<void()>, TAllocator>> qu(alloc);
//  volatile int64_t sum = 0;
//  auto start = std::chrono::high_resolution_clock::now();
//  for (int64_t i = 0; i < nItems; i++) {
//    qu.emplace(std::allocator_arg, alloc, [&sum]() { sum++; });
//  }
//
//  while (!qu.empty()) {
//    std::function<void()> f(std::move(qu.front()));
//    qu.pop();
//    f();
//  }
//  auto elapsed = std::chrono::high_resolution_clock::now() - start;
//  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//  printf("std::function with MemPool: %.3lf push&pop per second. Sum: %lld\n", nItems / nSec,
//    (long long)sum);
//}

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
      (int64_t(nThreads) * cnCacheEntryIncrements) / nSec, (long long)sum);

    thrs.clear();
  }
  _mm_free(const_cast<TCacheLineEntry*>(gpCacheLine));
}

const int64_t cLogsStart = 1; // 1000LL * 1000 * 1000;
const int64_t cnLogs = 1000LL * 1000 * 1000;

void BenchmarkLog2() {
  double sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const double x3 = double(cLogsStart + 3 * i);
    sum += std::log2(x3);
    const double x2 = double(cLogsStart + 7 * i);
    sum += std::log2(x2);
    const double x1 = double(cLogsStart + 17 * i);
    sum += std::log2(x1);
    const double x0 = double(cLogsStart + 37 * i);
    sum += std::log2(x0);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("std::log2: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void BenchmarkFpuLog2() {
  double sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const double x3 = double(cLogsStart + 3 * i);
    sum += SRPlat::SRLog2MulD(x3, 1);
    const double x2 = double(cLogsStart + 7 * i);
    sum += SRPlat::SRLog2MulD(x2, 1);
    const double x1 = double(cLogsStart + 17 * i);
    sum += SRPlat::SRLog2MulD(x1, 1);
    const double x0 = double(cLogsStart + 37 * i);
    sum += SRPlat::SRLog2MulD(x0, 1);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("FPU Log2: %.6lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum);
}

void BenchmarkLn() {
  double sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    const double x3 = double(cLogsStart + 3 * i);
    sum += std::log(x3);
    const double x2 = double(cLogsStart + 7 * i);
    sum += std::log(x2);
    const double x1 = double(cLogsStart + 17 * i);
    sum += std::log(x1);
    const double x0 = double(cLogsStart + 37 * i);
    sum += std::log(x0);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("Ln: %.3lf Ops/sec calculated %.6lf\n", cnLogs / nSec, sum / std::log(2.0));
}

void BenchmarkLog2Quads() {
  double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnLogs; i += 4) {
    sum0 += std::log2(double(i));
    sum1 += std::log2(double(i + 1));
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
  const __m128i cSseRoundingMantTblMask = _mm_set1_epi32(((1 << (cnLog2TblBits + 1)) - 1) << (20 - cnLog2TblBits - 1));
  const __m128i cSseRoundingBit = _mm_set1_epi32(1 << (20 - cnLog2TblBits - 1));
  const __m256i cAvxExp2YMask = _mm256_set1_epi64x(~((1ULL << (52 - cnLog2TblBits)) - 1));
  const __m256d cPlusBit = _mm256_castsi256_pd(_mm256_set1_epi64x(1ULL << (52 - cnLog2TblBits - 1)));
  double gLog2Table[(1 << cnLog2TblBits) + 1];
  double gPlusLog2Table[1 << cnLog2TblBits]; // plus |cnLog2TblBits|th highest bit
}

void InitLog2Table() {
  for (uint32_t i = 0; i < (1 << cnLog2TblBits); i++) {
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

  //const __m128i high32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), gHigh32Permute));
  const __m128 hiLane = _mm_castpd_ps(_mm256_extractf128_pd(x, 1));
  const __m128 loLane = _mm_castpd_ps(_mm256_castpd256_pd128(x));
  const __m128i high32 = _mm_castps_si128(_mm_shuffle_ps(loLane, hiLane, _MM_SHUFFLE(3, 1, 3, 1)));

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
    _mm256_slli_epi64(_mm256_cvtepu32_epi64(indexes), 52 - cnLog2TblBits)));
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
    const __m256d x = _mm256_set_pd(double(cLogsStart + 3 * i), double(cLogsStart + 7 * i), double(cLogsStart + 17 * i),
      double(cLogsStart + 37 * i));
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
    //const __m256d x = _mm256_set_pd(double(i + cLogsStart + 3), double(i + cLogsStart + 2), double(i + cLogsStart + 1),
    //  double(i + cLogsStart));
    const __m256d x = _mm256_set_pd(double(cLogsStart + 3 * i), double(cLogsStart + 7 * i), double(cLogsStart + 17 * i),
      double(cLogsStart + 37 * i));
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
    const __m256d x = _mm256_set_pd(double(cLogsStart + 3 * i), double(cLogsStart + 7 * i), double(cLogsStart + 17 * i),
      double(cLogsStart + 37 * i));
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
    const __m256d x = _mm256_set_pd(double(cLogsStart + 3 * i), double(cLogsStart + 7 * i), double(cLogsStart + 17 * i),
      double(cLogsStart + 37 * i));
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
  for (uint32_t i = 1; i <= std::thread::hardware_concurrency(); i++) {
    std::vector<std::thread> thrs;
    thrs.reserve(i);
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t j = 0; j < i; j++) {
      sums[2 * j] = _mm256_setzero_pd();
      thrs.emplace_back(AsyncLog2Vect, sums + 2 * j);
    }
    for (uint32_t j = 0; j < i; j++) {
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
    for (uint32_t j = 0; j < i; j++) {
      sums[8 * j] = 0;
      thrs.emplace_back(AsyncLn, sums + 8 * j);
    }
    for (uint32_t j = 0; j < i; j++) {
      thrs[j].join();
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    printf("%d threads Ln: %.3lf Ops/sec calculated %.6lf\n", int(i), cnLogs / nSec, sums[0]);
  }
}

//void TestNonconstMask() {
//  __m256i a = _mm256_set_epi64x(1, 2, 3, 4);
//  __m256i b = _mm256_set_epi64x(5, 6, 7, 8);
//  __m256i sum = _mm256_set1_epi64x(0);
//  for (int i = 0; i<16; i++) {
//    __m256i c = _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), i));
//    sum = _mm256_add_epi64(sum, c);
//  }
//  printf("%lld\n", sum.m256i_i64[0] + sum.m256i_i64[1] + sum.m256i_i64[2] + sum.m256i_i64[3]);
//}

void BitArrayTests() {
  //for(int i=0; i<128; i++) {
  //  volatile uint64_t quasiPow = SRPlat::SRMath::QuasiPowSqrt2(uint8_t(i));
  //  volatile int quasiLog = SRPlat::SRMath::QuasiCeilLogSqrt2(quasiPow);
  //  if(quasiLog != i) {
  //    printf("%d -> %d\n", i, quasiLog);
  //  }
  //}
  //for(int i=0; i<128; i++) {
  //  printf("%d -> %d\n", i, int(SRPlat::SRMath::CompressCapacity<2>(i)));
  //}
  //volatile uint64_t tIn = 65537;
  //volatile uint8_t tOut = SRPlat::SRMath::CeilLog2(tIn);
  SRPlat::SRBitArray ba(1);
  uint64_t nBits[2] = { 1, 0 };
  for (int i = 1; i <= 10000; i++) {
    ba.Add(i >> 1, i & 1);
    nBits[i & 1] += i >> 1;
  }
  uint64_t sum = 0;
  for (uint64_t i = 0; i < ba.Size() >> 2; i++) {
    sum += __popcnt16(ba.GetQuad(i));
  }
  for (uint64_t i = ba.Size()&(~3ui64); i < ba.Size(); i++) {
    sum += ba.GetOne(i) ? 1 : 0;
  }
  printf("Total %llu, expected1 %llu, actual1 %llu\n", ba.Size(), nBits[1], sum);
}

const uint16_t cnDoubleExps = 2048;
const size_t cnToBucket = 10 * 1000 * 1000;
double *gpBuckets = nullptr;

class FastRandomAdapter : public SRPlat::SRFastRandom {
public:
  static constexpr uint64_t min() { return 0; }
  static constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }
  uint64_t operator()() { return Generate64(); }
};

void AsyncBucketing() {
  //std::random_device rd;
  //SRPlat::SRFastRandom fr;
  //std::seed_seq seq{ rd(), rd(), rd(), rd() };
  //std::mt19937_64 rng(seq);
  FastRandomAdapter fra;
  std::normal_distribution<double> distr(0, 1e50);
  for (size_t i = 0; i < cnToBucket; i++) {
    const double val = distr(fra);
    const uint64_t ve = ((*reinterpret_cast<const uint64_t*>(&val)) >> 52) & 0x7ff;
    for (;;) {
      const double expected = *static_cast<volatile double*>(gpBuckets + ve);
      const double target = expected + val;
      if (*reinterpret_cast<const int64_t*>(&expected) == _InterlockedCompareExchange64(
        reinterpret_cast<int64_t*>(gpBuckets + ve),
        *reinterpret_cast<const int64_t*>(&target),
        *reinterpret_cast<const int64_t*>(&expected)))
      {
        break;
      }
    }
  }
}


void BenchmarkBucketing() {
  const uint32_t maxThreads = std::thread::hardware_concurrency();
  gpBuckets = reinterpret_cast<double*>(_mm_malloc(cnDoubleExps * sizeof(double), gCacheLineBytes));
  std::vector<std::thread> thrs;
  thrs.reserve(maxThreads + 1);

  for (uint32_t nThreads = 1; nThreads <= maxThreads; nThreads++) {
    std::memset(gpBuckets, 0, cnDoubleExps * sizeof(double));
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs.emplace_back(AsyncBucketing);
    }
    for (uint32_t i = 0; i < nThreads; i++) {
      thrs[i].join();
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    double sum = 0;
    for (uint32_t i = 0; i < cnDoubleExps; i++) {
      sum += gpBuckets[i];
    }
    printf("Bucketing double`s: %d threads give %.3lf Ops/sec, sum=%.16e\n", (int)nThreads,
      (nThreads * cnToBucket) / nSec, (double)sum);

    thrs.clear();
  }
  _mm_free(gpBuckets);
}

template<int taError> class MyException : public std::exception {
  using std::exception::exception;
  virtual const char* what() const throw() override {
    printf("It's %d\n", taError);
    return std::exception::what();
  }
};

template<int maxErr> void selectThrow(int myErr) {
  if (myErr == maxErr) {
    throw MyException<maxErr>();
  }
  return selectThrow<maxErr - 1>(myErr);
}

template<> void selectThrow<0>(int) {
}

template<typename... taOthers> void throwVariadic(int) {
}

template<int taCur, int... taOthers> void throwVariadic(int errCode) {
  if (taCur == errCode) {
    throw MyException<taCur>();
  }
  return throwVariadic<taOthers...>(errCode);
}

void checkError(int errCode) {
  throwVariadic<1, 3, 7, 15, 25>(errCode);
}

void testSet1() {
  __m256i sum = _mm256_setzero_si256();
  for (int64_t i = 0; i < 1000; i++) {
    sum = _mm256_add_epi64(sum, _mm256_set1_epi64x(1023));
  }
  for (int8_t i = 0; i <= 3; i++) {
    printf(" %lld", sum.m256i_i64[i]);
  }
}

void testErrCodesToExceptions() {
  int a = rand();
  checkError(a);
  if (a != 0) {
    selectThrow <100>(a);
  }
}

////// Set To Bit Quad benchmarks
__m256i __vectorcall StbqSet(const uint8_t bitQuad) {
  return _mm256_set_epi64x(-int64_t(bitQuad >> 3), -int64_t((bitQuad >> 2) & 1), -int64_t((bitQuad >> 1) & 1),
    -int64_t(bitQuad & 1));
}

__m256i __vectorcall StbqPdep(const uint8_t bitQuad) {
  __m128i source = _mm_cvtsi32_si128(_pdep_u32(bitQuad, 0b10000000100000001000000010000000));
  return _mm256_cvtepi8_epi64(source);
}

uint32_t gStbqTable[16];

void InitStbqTable() {
  for (int8_t i = 0; i < 16; i++) {
    int8_t comps[4];
    comps[0] = -(i & 1);
    comps[1] = -((i >> 1) & 1);
    comps[2] = -((i >> 2) & 1);
    comps[3] = -(i >> 3);
    uint32_t val = *reinterpret_cast<uint32_t*>(comps);
    gStbqTable[i] = val;
  }
}

__m256i __vectorcall StbqTbl(const uint8_t bitQuad) {
  __m128i source = _mm_cvtsi32_si128(gStbqTable[bitQuad]);
  return _mm256_cvtepi8_epi64(source);
}

const int64_t cnStbqs = 1000 * 1000i64 * 1000;

void BenchmarkStbqSet() {
  __m256i sums = _mm256_setzero_si256();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnStbqs; i++) {
    const __m256i res = StbqSet(i & 0xf);
    sums = _mm256_add_epi64(sums, res);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  int64_t sum = sums.m256i_i64[0] + sums.m256i_i64[1] + sums.m256i_i64[2] + sums.m256i_i64[3];
  printf("STBQ via set: %.3lf Ops/sec calculated %" PRId64 "\n", cnStbqs / nSec, sum);
}

void BenchmarkStbqPdep() {
  __m256i sums = _mm256_setzero_si256();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnStbqs; i++) {
    const __m256i res = StbqPdep(i & 0xf);
    sums = _mm256_add_epi64(sums, res);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  int64_t sum = sums.m256i_i64[0] + sums.m256i_i64[1] + sums.m256i_i64[2] + sums.m256i_i64[3];
  printf("STBQ via PDEP: %.3lf Ops/sec calculated %" PRId64 "\n", cnStbqs / nSec, sum);
}

void BenchmarkStbqTbl() {
  __m256i sums = _mm256_setzero_si256();
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 1; i <= cnStbqs; i++) {
    const __m256i res = StbqTbl(i & 0xf);
    sums = _mm256_add_epi64(sums, res);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  int64_t sum = sums.m256i_i64[0] + sums.m256i_i64[1] + sums.m256i_i64[2] + sums.m256i_i64[3];
  printf("STBQ via table: %.3lf Ops/sec calculated %" PRId64 "\n", cnStbqs / nSec, sum);
}

void VectorPush() {
  SRPlat::SRFastArray<double, true> fa1(10);
  SRPlat::SRFastArray<double, true> fa2(fa1);
  std::vector<SRPlat::SRFastArray<double, true>> v;
  v.reserve(1);
  v.emplace_back(10);
  v.emplace_back(10);
}

//int __cdecl main() {
//  //std::atomic<double> test1;
//  //bool test2 = test1.is_lock_free();
//  //std::cout << test2 << std::endl;
//
//  //BenchmarkFunctor();
//  //BenchmarkObject();
//  //BenchmarkMSVCpp();
//  //BenchmarkTemplate();
//  //BenchmarkMacro();
//  //BenchmarkEmpty();
//  //BenchmarkAtomic();
//  //MultibenchMemory();
//
//  //BenchmarkStdFunctionMemPool();
//  //BenchmarkStdFunctionStdAlloc();
//  //BenchmarkSubtask();
//  //BenchmarkSmallQueue();
//  //BenchmarkLambda();
//
//  //BenchmarkCacheLine();
//  //BenchmarkLog2Quads();
//  //BenchmarkLog2VectInl();
//  //BenchmarkLnThreads();
//  //BenchmarkLog2VectThreads();
//
//  //InitLog2Table();
//  //BenchmarkLog2tblPlus();
//  //BenchmarkLog2tblPrec();
//  //BenchmarkLog2tblVect();
//  //BenchmarkLog2Vect();
//  //BenchmarkLn();
//  //BenchmarkLog2();
//  //BenchmarkFpuLog2();
//
//  //BenchmarkBucketing();
//
//  InitStbqTable();
//  BenchmarkStbqTbl();
//  BenchmarkStbqSet();
//
//  //BenchmarkStbqPdep();
//
//  return 0;
//}
