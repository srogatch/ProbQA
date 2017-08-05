// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CECreateQuizOperation.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/Interface/PqaErrors.h"

namespace ProbQA {

class CECreateQuizOpBase {
public: // constants
  enum class Operation : uint8_t {
    None = 0,
    Start = 1,
    Resume = 2
  };
public: // variables
  PqaError& _err;

public: // methods
  explicit CECreateQuizOpBase(PqaError& err);
  virtual Operation GetCode() = 0;
  virtual ~CECreateQuizOpBase();
};

class CECreateQuizStart : public CECreateQuizOpBase {
public: //methods
  explicit CECreateQuizStart(PqaError& err);
  virtual Operation GetCode() override final;
};

template<typename taNumber> class CECreateQuizResume : public CECreateQuizOpBase {
  static constexpr uint32_t _cScalarCacheUsageBytes = 1024;
  
public: // variables
  const TPqaId _nAnswered;
  const AnsweredQuestion* const _pAQs;

private: //methods
  static std::enable_if_t<SRPlat::SRSimd::_cNBytes % sizeof(taNumber) == 0, uint32_t> CalcVectsInCache();

public: //methods
  explicit CECreateQuizResume(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs);
  virtual Operation GetCode() override final;
  void ApplyAnsweredQuestions(CpuEngine<taNumber> *pCe, CEQuiz<taNumber> *pQuiz);
};

} // namespace ProbQA
