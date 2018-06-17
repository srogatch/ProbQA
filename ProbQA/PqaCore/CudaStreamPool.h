// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

class CudaStreamPool;

// Lifetime must not exceed the lifetime of the pool.
class CudaStream {
  cudaStream_t _stream;
  CudaStreamPool *_pPool;

public:
  CudaStream(cudaStream_t stream, CudaStreamPool *pPool) : _stream(stream), _pPool(pPool) { }
  ~CudaStream();
  CudaStream(CudaStream &&fellow);
  CudaStream& operator=(CudaStream &&fellow);
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t Get() const { return _stream; }
};

class CudaStreamPool {
private: // types
  typedef SRPlat::SRSpinSync<(1<<5)> TSync;
private: // variables
  std::vector<cudaStream_t> _pool;
  TSync _sync;
  const bool _bBlocking;

public: // methods
  explicit CudaStreamPool(const bool blocking);
  ~CudaStreamPool();
  CudaStream Acquire();
  void Release(cudaStream_t stream);
};

} // namespace ProbQA
