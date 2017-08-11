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

  BenchmarkCacheLine();

  return 0;
}

