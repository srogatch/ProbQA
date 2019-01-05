// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CudaException.h"
#include "../PqaCore/CudaMacros.h"
#include "../PqaCore/Interface/CudaMain.h"

namespace ProbQA {

template<typename T> class  CudaArray {
  T *_d_p;

private: // methods
  static void Destroy(void* d_p) {
    CUDA_MUST(cudaFree(d_p));
  }

public: // methods
  explicit CudaArray() {
    _d_p = nullptr;
  }

  explicit CudaArray(const int64_t nItems) {
    CUDA_MUST(cudaMalloc(&_d_p, sizeof(T)*nItems));
  }
  
  CudaArray(const CudaArray&) = delete;
  CudaArray& operator=(const CudaArray&) = delete;

  CudaArray(CudaArray&& source) noexcept : _d_p(source._d_p) {
    source._d_p = nullptr;
  }

  CudaArray& operator=(CudaArray&& source) {
    if (this != &source) {
      Destroy(_d_p);
      _d_p = source._d_p;
      source._d_p = nullptr;
    }
    return *this;
  }

  ~CudaArray() {
    try {
      Destroy(_d_p);
    }
    catch (CudaException &ex) {
      SRPlat::SRDefaultLogger::Get()->Log(SRPlat::ISRLogger::Severity::Error, ex.GetMsg());
    }
  }

  T* Get() const { return _d_p; }

  void EarlyRelease() {
    Destroy(_d_p);
    _d_p = nullptr;
  }
};

} // namespace ProbQA
