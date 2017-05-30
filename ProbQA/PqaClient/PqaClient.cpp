// PqaClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int main()
{
  const int64_t nIterations = 10 * 1000 * 1000;
  HANDLE hEvent = CreateEvent(nullptr, true, true, nullptr);
  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < nIterations; i++) {
    WaitForSingleObject(hEvent, INFINITE);
  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double nSec = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  printf("%.3lf Ops/sec\n", nIterations / nSec);
  return 0;
}

