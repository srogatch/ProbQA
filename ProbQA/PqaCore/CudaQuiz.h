// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseQuiz.h"
#include "../PqaCore/CudaMemPool.h"

namespace ProbQA {

template<typename taNumber> class CudaEngine;
class BaseCpuEngine;

template<typename taNumber> class CudaQuiz : public BaseQuiz {
public:
  typedef int64_t TExponent;

private: // variables
  CudaMPArray<uint8_t> _storage;
  uint8_t *_pQAsked;
  TExponent *_pExponents;
  taNumber *_pPriorMants;

public: // methods
  CudaQuiz(CudaEngine<taNumber> *pEngine);
  ~CudaQuiz();

  uint8_t* GetQAsked() const { return _pQAsked; }
  TExponent* GetTlhExps() const { return _pExponents; }
  taNumber* GetPriorMants() const { return _pPriorMants; }

  virtual PqaError RecordAnswer(const TPqaId iAnswer) override final;
};

} // namespace ProbQA
