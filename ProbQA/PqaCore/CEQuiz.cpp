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
  auto& memPool = _pEngine->GetMemPool();
  //SRSmartMPP<decltype(memPool), __m256i> smppIsQAsked(memPool, (dims._nQuestions + 255) >> 8);
  _isQAsked = static_cast<__m256i*>(memPool.AllocMem((dims._nQuestions+7)>>3));
  SRBitHelper::FillZero<true>(_isQAsked, dims._nQuestions);
  _pTargProbs = static_cast<taNumber*>(memPool.AllocMem(sizeof(taNumber) * dims._nTargets));
}

template<typename taNumber> CEQuiz<taNumber>::~CEQuiz() {
  const EngineDimensions& dims = _pEngine->GetDims();
  auto& memPool = _pEngine->GetMemPool();
  memPool.ReleaseMem(_pTargProbs, sizeof(taNumber) * dims._nTargets);
  //memPool.ReleaseMem();
}

template class CEQuiz<DoubleNumber>;

} // namespace ProbQA
