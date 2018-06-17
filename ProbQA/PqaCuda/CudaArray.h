// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCuda/Interface/PqaCuda.h"
#include "../PqaCuda/CudaException.h"

namespace PqaCuda {

class  CudaArray {
  void *_d_p;

private: // methods
  static void Destroy(void* d_p);

public: // methods
  explicit CudaArray(const int64_t nBytes);
  
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
};

} // namespace PqaCuda
