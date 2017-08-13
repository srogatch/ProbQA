// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEUpdatePriorsTask.decl.h"

namespace ProbQA {

template<typename taNumber> inline CEUpdatePriorsTask<taNumber>::CEUpdatePriorsTask(CpuEngine<taNumber> &engine,
  CEQuiz<taNumber> &quiz, const TPqaId nAnswered, const AnsweredQuestion* const pAQs, const uint32_t nVectsInCache)
  : CEBaseTask(engine), _pQuiz(&quiz), _nAnswered(nAnswered), _pAQs(pAQs), _nVectsInCache(nVectsInCache)
{ }

} // namespace ProbQA
