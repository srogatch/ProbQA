#include "stdafx.h"
#include "../PqaCore/CudaQuiz.h"
#include "../PqaCore/CudaEngine.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/CudaEngineGpu.cuh"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CudaQuiz<taNumber>::CudaQuiz(CudaEngine<taNumber> *pEngine) : BaseQuiz(pEngine) {
  const TPqaId nTargets = pEngine->GetDims()._nTargets;
  const TPqaId nQuestions = pEngine->GetDims()._nQuestions;
  const size_t nBitVects = SRSimd::VectsFromBits(nQuestions);
  _storage = CudaArray<uint8_t, true>(SRSimd::_cNBytes * (nBitVects + /* alignment */ 1)
    + SRSimd::GetPaddedBytes(nTargets * sizeof(taNumber))
    + SRSimd::GetPaddedBytes(nTargets * sizeof(TExponent))
  );
  _pQAsked = static_cast<__m256i*>(SRSimd::AlignPtr(_storage.Get(), SRSimd::_cNBytes));
  _pPriorMants = reinterpret_cast<taNumber*>(_pQAsked + nBitVects);
  _pExponents = static_cast<TExponent*>(SRSimd::AlignPtr(_pPriorMants + nTargets));
}

template<typename taNumber> CudaQuiz<taNumber>::~CudaQuiz() {
}

template<typename taNumber> PqaError CudaQuiz<taNumber>::RecordAnswer(const TPqaId iAnswer) {
  _answers.emplace_back(_activeQuestion, iAnswer);
  SRBitHelper::Set(GetQAsked(), _activeQuestion);
  _activeQuestion = cInvalidPqaId;

  CudaEngine<taNumber> *pEngine = static_cast<CudaEngine<taNumber>*>(GetBaseEngine());
  RecordAnswerKernel<taNumber> rak;
  rak._iAnswer = iAnswer;
  rak._iQuestion = _answers.back()._iQuestion;
  rak._nAnswers = pEngine->GetDims()._nAnswers;
  rak._nQuestions = pEngine->GetDims()._nQuestions;
  rak._nTargets = pEngine->GetDims()._nTargets;
  rak._pPriorMants = _pPriorMants;
  rak._pmD = pEngine->GetMD();
  rak._psA = pEngine->GetSA();
  {
    CudaDeviceLock cdl = CudaMain::SetDevice(pEngine->GetDevice());
    CudaStream cuStr = pEngine->GetCspNb().Acquire();
    SRRWLock<false> rwl(pEngine->GetRws());
    rak.Run(pEngine->GetKlc(), cuStr.Get());
    CUDA_MUST(cudaGetLastError());
    CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
  }
  return PqaError();
}

//// Instantiations
template class CudaQuiz<float>;

} // namespace ProbQA
