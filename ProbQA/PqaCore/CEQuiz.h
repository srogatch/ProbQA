// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEQuiz.decl.h"
#include "../PqaCore/CpuEngine.h"

namespace ProbQA {

//////////////////////////////// CEBaseQuiz implementation /////////////////////////////////////////////////////////////

inline CEBaseQuiz::CEBaseQuiz(BaseCpuEngine *pEngine) : _pEngine(pEngine) {
  const EngineDimensions& dims = _pEngine->GetDims();
  const size_t nQuestions = SRPlat::SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = _pEngine->GetMemPool();

  // First allocate all the memory so to revert if anything fails.
  SRPlat::SRSmartMPP<__m256i> smppIsQAsked(memPool, SRPlat::SRSimd::VectsFromBits(nQuestions));
  SRPlat::SRSmartMPP<TExponent> smppExponents(memPool, nTargets);

  // As all the memory is allocated, safely proceed with finishing construction of CEBaseQuiz object.
  _pTlhExps = smppExponents.Detach();
  _isQAsked = smppIsQAsked.Detach();
}

inline CEBaseQuiz::~CEBaseQuiz() {
  const EngineDimensions& dims = GetBaseEngine()->GetDims();
  const size_t nQuestions = SRPlat::SRCast::ToSizeT(dims._nQuestions);
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = GetBaseEngine()->GetMemPool();
  //NOTE: engine dimensions must not change during lifetime of the quiz because below we must provide the same number
  //  of targets and questions.
  memPool.ReleaseMem(_pTlhExps, sizeof(*_pTlhExps) * nTargets);
  memPool.ReleaseMem(_isQAsked, SRPlat::SRBitHelper::GetAlignedSizeBytes(nQuestions));
}

//////////////////////////////// CEQuiz implementation /////////////////////////////////////////////////////////////////

template<typename taNumber> inline CpuEngine<taNumber>* CEQuiz<taNumber>::GetEngine() const {
  return static_cast<CpuEngine<taNumber>*>(GetBaseEngine());
}

template<typename taNumber> CEQuiz<taNumber>::CEQuiz(CpuEngine<taNumber> *pEngine) : CEBaseQuiz(pEngine) {
  const EngineDimensions& dims = pEngine->GetDims();
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = pEngine->GetMemPool();

  // First allocate all the memory so to revert if anything fails.
  SRPlat::SRSmartMPP<taNumber> smppMantissas(memPool, nTargets);

  // As all the memory is allocated, safely proceed with finishing construction of CEQuiz object.
  _pTlhMants = smppMantissas.Detach();
}

template<typename taNumber> CEQuiz<taNumber>::~CEQuiz() {
  const EngineDimensions& dims = GetBaseEngine()->GetDims();
  const size_t nTargets = SRPlat::SRCast::ToSizeT(dims._nTargets);
  auto& memPool = GetBaseEngine()->GetMemPool();
  //NOTE: engine dimensions must not change during lifetime of the quiz because below we must provide the same number
  //  of targets.
  memPool.ReleaseMem(_pTlhMants, sizeof(*_pTlhMants) * nTargets);
}

} // namespace ProbQA
