// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaErrors.h"

namespace ProbQA {

template<typename taNumber> class CEListTopTargetsAlgorithm {
public: // constants
  static constexpr uint32_t _cnRadixSortBuckets = 256;

private: // variables
  CpuEngine<taNumber> *const PTR_RESTRICT _pEngine;
  const CEQuiz<taNumber> *const PTR_RESTRICT _pQuiz;
  RatedTarget *const PTR_RESTRICT _pDest;
  PqaError &PTR_RESTRICT _err;
  const TPqaId _maxCount;

public: // variables
  const TPqaId _nTargets;
  const SRPlat::SRThreadCount _nWorkers;

public:
  explicit CEListTopTargetsAlgorithm(PqaError &PTR_RESTRICT err, CpuEngine<taNumber> &PTR_RESTRICT engine,
    const CEQuiz<taNumber> &PTR_RESTRICT quiz, const TPqaId maxCount, RatedTarget *PTR_RESTRICT pDest);

  TPqaId RunHeapifyBased();
  TPqaId RunRadixSortBased();
};

} // namespace ProbQA

