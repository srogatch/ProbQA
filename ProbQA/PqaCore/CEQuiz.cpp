// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CEQuiz<taNumber>::CEQuiz(CpuEngine<taNumber> *pEngine)
  : _pEngine(pEngine)
{
  const EngineDimensions& dims = _pEngine->GetDims();
  const size_t nQuestions = SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRCast::ToSizeT(dims._nTargets);
  typedef CpuEngine<taNumber>::TMemPool TMemPool;
  TMemPool& memPool = _pEngine->GetMemPool();

  // First allocate all the memory so to revert if anything fails.
  SRSmartMPP<TMemPool, __m256i> smppIsQAsked(memPool, SRSimd::VectsFromBits(nQuestions));
  SRSmartMPP<TMemPool, taNumber> smppMantissas(memPool, nTargets);
  SRSmartMPP<TMemPool, TExponent> smppExponents(memPool, nTargets);

  // As all the memory is allocated, safely proceed with finishing construction of CEQuiz object.
  _pTlhExps = smppExponents.Detach();
  _pTlhMants = smppMantissas.Detach();
  _isQAsked = smppIsQAsked.Detach();
}

template<typename taNumber> CEQuiz<taNumber>::~CEQuiz() {
  const EngineDimensions& dims = _pEngine->GetDims();
  const size_t nQuestions = SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRCast::ToSizeT(dims._nTargets);
  auto& memPool = _pEngine->GetMemPool();
  //NOTE: engine dimensions must not change during lifetime of the quiz because below we must provide the same number
  //  of targets and questions.
  memPool.ReleaseMem(_pTlhExps, sizeof(*_pTlhExps) * nTargets);
  memPool.ReleaseMem(_pTlhMants, sizeof(*_pTlhMants) * nTargets);
  memPool.ReleaseMem(_isQAsked, SRBitHelper::GetAlignedSizeBytes(nQuestions));
}

template class CEQuiz<DoubleNumber>;

} // namespace ProbQA
