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
  typedef CpuEngine<taNumber>::TMemPool TMemPool;
  TMemPool& memPool = _pEngine->GetMemPool();
  SRSmartMPP<TMemPool, __m256i> smppIsQAsked(memPool, (dims._nQuestions + 255) >> 8);
  SRSmartMPP<TMemPool, taNumber> smppTargProbs(memPool, dims._nTargets);
  SRBitHelper::FillZero<true>(smppIsQAsked.Get(), dims._nQuestions);
  _isQAsked = smppIsQAsked.Detach();
  _pTargProbs = smppTargProbs.Detach();
}

template<typename taNumber> CEQuiz<taNumber>::~CEQuiz() {
  const EngineDimensions& dims = _pEngine->GetDims();
  auto& memPool = _pEngine->GetMemPool();
  memPool.ReleaseMem(_pTargProbs, sizeof(taNumber) * dims._nTargets);
  memPool.ReleaseMem(_isQAsked, SRBitHelper::GetAlignedSizeBytes(dims._nQuestions));
}

template class CEQuiz<DoubleNumber>;

} // namespace ProbQA
