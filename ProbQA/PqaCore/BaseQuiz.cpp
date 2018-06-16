// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseQuiz.h"

using namespace SRPlat;

namespace ProbQA {

BaseQuiz::BaseQuiz(BaseEngine *pEngine) : _pEngine(pEngine) {
  const EngineDimensions& dims = _pEngine->GetDims();
  const size_t nQuestions = SRPlat::SRCast::ToSizeT(dims._nQuestions);
  SRMemTotal mtCommon;
  SRMemItem<__m256i> miIsQAsked(SRPlat::SRSimd::VectsFromBits(nQuestions), SRPlat::SRMemPadding::Both, mtCommon);
  // First allocate all the memory so to revert if anything fails.
  SRSmartMPP<uint8_t> commonBuf(_pEngine->GetMemPool(), mtCommon._nBytes);
  // Must be the first memory block, because it's used for releasing the memory
  _isQAsked = miIsQAsked.Ptr(commonBuf);
  // As all the memory is allocated, safely proceed with finishing construction of CEBaseQuiz object.
  commonBuf.Detach();
}

BaseQuiz::~BaseQuiz() {

}

} // namespace ProbQA
