// PqaClient.cpp : Defines the entry point for the console application.
//

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

int __cdecl main() {
  //BenchmarkFunctor();
  //BenchmarkObject();
  //BenchmarkMSVCpp();
  //BenchmarkTemplate();
  //BenchmarkMacro();
  //BenchmarkEmpty();
  BenchmarkAtomic();
  return 0;
}

