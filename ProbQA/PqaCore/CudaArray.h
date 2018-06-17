// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CudaException.h"
#include "../PqaCore/CudaMacros.h"
#include "../PqaCore/Interface/CudaMain.h"

namespace ProbQA {

template<typename T, bool taUnified> class  CudaArray {
  T *_d_p;

private: // methods
  static void Destroy(void* d_p) {
    CUDA_MUST(cudaFree(d_p));
  }

public: // methods
  explicit CudaArray(const int64_t nItems) {
    if (taUnified) {
      CUDA_MUST(cudaMallocManaged(&_d_p, sizeof(T)*nItems));
    }
    else {
      CUDA_MUST(cudaMalloc(&_d_p, sizeof(T)*nItems));
    }
  }
  
  CudaArray(const CudaArray&) = delete;
  CudaArray& operator=(const CudaArray&) = delete;

  CudaArray(CudaArray&& source) : _d_p(source._d_p) {
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

  // Pass dstDevice=cudaCpuDeviceId for copying the data to CPU memory.
  void Prefetch(const cudaStream_t stream, const int64_t iFirst, const int64_t nItems, int destDevice) {
    if constexpr(!taUnified) {
      SRPlat::SRException(SRPlat::SRString::MakeUnowned(SR_FILE_LINE "Requested prefetch on non-unified memory."))
        .ThrowMoving();
    }
    CUDA_MUST(cudaMemPrefetchAsync(_d_p + iFirst, sizeof(T)*nItems, destDevice, stream));
  }
};

} // namespace ProbQA
