// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CudaMacros.h"

namespace ProbQA {

class CudaMemPool;

template<typename T> class  CudaMPArray {
  T *_d_p;
  CudaMemPool *_pMp;
  size_t _nBytes;

private: // methods
  static void Destroy(void* d_p, CudaMemPool& mp, size_t nBytes);

public: // methods
  explicit CudaMPArray(CudaMemPool &cmp) : _pMp(&cmp), _d_p(nullptr), _nBytes(0) {
  }
  explicit CudaMPArray(CudaMemPool &cmp, const int64_t nItems);

  CudaMPArray(const CudaMPArray&) = delete;
  CudaMPArray& operator=(const CudaMPArray&) = delete;

  CudaMPArray(CudaMPArray&& source) : _d_p(source._d_p), _pMp(source._pMp), _nBytes(source._nBytes) {
    source._d_p = nullptr;
  }

  CudaMPArray& operator=(CudaMPArray&& source) {
    if (this != &source) {
      Destroy(_d_p, *_pMp, _nBytes);
      _d_p = source._d_p;
      _pMp = source._pMp;
      _nBytes = source._nBytes;
      source._d_p = nullptr;
    }
    return *this;
  }

  ~CudaMPArray() {
    try {
      Destroy(_d_p, *_pMp, _nBytes);
    }
    catch (CudaException &ex) {
      SRPlat::SRDefaultLogger::Get()->Log(SRPlat::ISRLogger::Severity::Error, ex.GetMsg());
    }
  }

  T* Get() const { return _d_p; }

  void EarlyRelease() {
    Destroy(_d_p, *pMp, _nBytes);
    _d_p = nullptr;
  }
};

class CudaMemPool {
  typedef SRPlat::SRSpinSync< 1 << 5 > TSync;
  typedef std::unordered_map<size_t, std::vector<void*>> TPool;
  const size_t _maxBytes;
  std::atomic<size_t> _totBytes;
  TSync _sync;
  TPool _pool;
public:
  explicit CudaMemPool(size_t maxBytes) : _totBytes(0), _maxBytes(maxBytes) {
  }
  ~CudaMemPool() {
    FreeAllChunks();
  }
  void* Acquire(const size_t nBytes) {
    SRPlat::SRLock<TSync> sl(_sync);
    TPool::iterator it = _pool.find(nBytes);
    if (it == _pool.end() || it->second.size() == 0) {
      sl.EarlyRelease();
      void *d_p;
      CUDA_MUST(cudaMalloc(&d_p, nBytes));
      _totBytes.fetch_add(nBytes, std::memory_order_relaxed);
      return d_p;
    }
    void *res = it->second.back();
    it->second.pop_back();
    return res;
  }

  void Release(void *d_p, const size_t nBytes) {
    if (d_p == nullptr) {
      return;
    }
    if (nBytes + _totBytes.load(std::memory_order_relaxed) > _maxBytes) {
      CUDA_MUST(cudaFree(d_p));
      return;
    }

    SRPlat::SRLock<TSync> sl(_sync);
    TPool::iterator it = _pool.find(nBytes);
    if (it == _pool.end()) {
      std::vector<void*> items;
      items.push_back(d_p);
      _pool.emplace(nBytes, std::move(items));
    }
    else {
      it->second.push_back(d_p);
    }
  }

  void FreeAllChunks() {
    TPool toDel;
    {
      SRPlat::SRLock<TSync> sl(_sync);
      toDel = std::move(_pool);
    }
    for (TPool::iterator it = toDel.begin(); it != toDel.end(); it++) {
      for (size_t i = 0; i < it->second.size(); it++) {
        CUDA_MUST(cudaFree(it->second[i]));
      }
    }
  }
};

template<typename T> CudaMPArray<T>::CudaMPArray(CudaMemPool &cmp, const int64_t nItems) : _pMp(&cmp) {
  _nBytes = sizeof(T)*nItems;
  _d_p = static_cast<T*>(_pMp->Acquire(_nBytes));
}

template<typename T> void CudaMPArray<T>::Destroy(void* d_p, CudaMemPool& mp, size_t nBytes) {
  mp.Release(d_p, nBytes);
}

} // namespace ProbQA
