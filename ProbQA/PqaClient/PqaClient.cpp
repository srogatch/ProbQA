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
    gCounter.store(0, std::memory_order_release);
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
    gpdInput[i] = fr.Generate() / (double(fr.Generate()) + 1);
    //gpdInput[i] = fr.SimdGenerate() / (double(fr.SimdGenerate()) + 1);
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
    const __m256i loaded = _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(pSrc));
    const __m256d dividend = *reinterpret_cast<const __m256d*>(&loaded);
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
  memcpy(gpdOutput, gpdInput, sizeof(double)*cnDoubles);
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

int __cdecl main() {
  //BenchmarkFunctor();
  //BenchmarkObject();
  //BenchmarkMSVCpp();
  //BenchmarkTemplate();
  //BenchmarkMacro();
  //BenchmarkEmpty();
  //BenchmarkAtomic();
  gpdInput = (double*)_mm_malloc(cnDoubles * sizeof(double), sizeof(__m256d));
  gpdOutput  = (double*)_mm_malloc(cnDoubles * sizeof(double), sizeof(__m256d));
  // BenchmarkRandFill();

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

  _mm_free(gpdInput);
  _mm_free(gpdOutput);
  return 0;
}

