// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/BaseCpuEngine.h"
#include "../PqaCore/BaseQuiz.h"

namespace ProbQA {

class CEBaseQuiz : public BaseQuiz {
public: // types
  typedef int64_t TExponent;

private:
  // Exponents for the target likelyhoods: x[i] = _pTlhs[i] * pow(2, _pExps[i])
  TExponent *_pTlhExps;

protected: // methods
  inline explicit CEBaseQuiz(BaseCpuEngine *pEngine);
  inline ~CEBaseQuiz() override;

public: // methods
  TExponent* GetTlhExps() const { return _pTlhExps; }
};

template<typename taNumber> class CEQuiz : public CEBaseQuiz {
  // For precision and to avoid underflow, mantissas and exponents are stored separately.
  // Priors must be usually normalized, except for short periods of updating them.
  taNumber *_pPriorMants;

public: // methods
  explicit CEQuiz(CpuEngine<taNumber> *pEngine);
  ~CEQuiz() override final;
  taNumber* GetPriorMants() const { return _pPriorMants; }
  CpuEngine<taNumber>* GetEngine() const;
  PqaError RecordAnswer(const TPqaId iAnswer) override final;
};

} // namespace ProbQA
