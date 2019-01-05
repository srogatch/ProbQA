// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaStreamPool.h"
#include "../PqaCore/CudaMacros.h"

using namespace SRPlat;

namespace ProbQA {

/////////////////////////////////// CudaStream class implementation ////////////////////////////////////////////////////

CudaStream::~CudaStream() {
  if (_pPool != nullptr) {
    _pPool->Release(_stream);
  }
}

CudaStream::CudaStream(CudaStream &&fellow) noexcept : _stream(fellow._stream), _pPool(fellow._pPool) {
  fellow._stream = nullptr;
  fellow._pPool = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream &&fellow) {
  if (this != &fellow) {
    if (_pPool != nullptr) {
      _pPool->Release(_stream);
    }
    _stream = fellow._stream;
    _pPool = fellow._pPool;
    fellow._stream = nullptr;
    fellow._pPool = nullptr;
  }
  return *this;
}

/////////////////////////////////// CudaStreamPool class implementation ////////////////////////////////////////////////

CudaStreamPool::CudaStreamPool(const bool bBlocking) : _bBlocking(bBlocking) {
}

CudaStreamPool::~CudaStreamPool() {
  for (cudaStream_t stream : _pool) {
    CUDA_MUST(cudaStreamDestroy(stream));
  }
}

CudaStream CudaStreamPool::Acquire() {
  {
    SRLock<TSync> sl(_sync);
    if (_pool.size() >= 1) {
      cudaStream_t stream = _pool.back();
      _pool.pop_back();
      sl.EarlyRelease();
      return CudaStream(stream, this);
    }
  }
  cudaStream_t stream;
  // Let it work concurrently with stream 0
  CUDA_MUST(cudaStreamCreateWithFlags(&stream, _bBlocking ? 0 : cudaStreamNonBlocking));
  return CudaStream(stream, this);
}

void CudaStreamPool::Release(cudaStream_t stream) {
  assert(stream != nullptr);
  SRLock<TSync> sl(_sync);
  _pool.push_back(stream);
}

} // namespace ProbQA
