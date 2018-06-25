#include "stdafx.h"
#include "../PqaCore/CudaQuiz.h"
#include "../PqaCore/CudaEngine.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/CudaEngineGpu.cuh"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CudaQuiz<taNumber>::CudaQuiz(CudaEngine<taNumber> *pEngine) : BaseQuiz(pEngine),
  _storage(pEngine->GetCuMp())
{
  const TPqaId nTargets = pEngine->GetDims()._nTargets;
  const TPqaId nQuestions = pEngine->GetDims()._nQuestions;
  const size_t nBitVects = SRSimd::VectsFromBits(nQuestions);
  _storage = CudaMPArray<uint8_t>(pEngine->GetCuMp(),
    SRSimd::_cNBytes * (nBitVects + /* alignment */ 1)
    + SRSimd::GetPaddedBytes(nTargets * sizeof(taNumber))
    + SRSimd::GetPaddedBytes(nTargets * sizeof(TExponent))
  );
  _pQAsked = static_cast<uint8_t*>(SRSimd::AlignPtr(_storage.Get()));
  _pPriorMants = reinterpret_cast<taNumber*>(reinterpret_cast<__m256i*>(_pQAsked) + nBitVects);
  _pExponents = static_cast<TExponent*>(SRSimd::AlignPtr(_pPriorMants + nTargets));
}

template<typename taNumber> CudaQuiz<taNumber>::~CudaQuiz() {
}

template<typename taNumber> PqaError CudaQuiz<taNumber>::RecordAnswer(const TPqaId iAnswer) {
  CudaEngine<taNumber> *pEngine = static_cast<CudaEngine<taNumber>*>(GetBaseEngine());
  RecordAnswerKernel<taNumber> rak;
  rak._iAnswer = iAnswer;
  rak._iQuestion = _activeQuestion;
  rak._nAnswers = pEngine->GetDims()._nAnswers;
  rak._nQuestions = pEngine->GetDims()._nQuestions;
  rak._nTargets = pEngine->GetDims()._nTargets;
  rak._pPriorMants = _pPriorMants;
  rak._pmD = pEngine->GetMD();
  rak._psA = pEngine->GetSA();
  rak._pTargetGaps = pEngine->DevTargetGaps();
  {
    CudaDeviceLock cdl = CudaMain::SetDevice(pEngine->GetDevice());
    CudaStream cuStr = pEngine->GetCspNb().Acquire();

    uint8_t cpuByte;
    uint8_t *pDevByte = _pQAsked + (_activeQuestion >> 3);
    CUDA_MUST(cudaMemcpyAsync(&cpuByte, pDevByte, 1, cudaMemcpyDeviceToHost, cuStr.Get()));
    CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
    cpuByte |= (1 << (_activeQuestion & 7));
    CUDA_MUST(cudaMemcpyAsync(pDevByte, &cpuByte, 1, cudaMemcpyHostToDevice, cuStr.Get()));

    SRRWLock<false> rwl(pEngine->GetRws());
    rak.Run(pEngine->GetKlc(), cuStr.Get());
    CUDA_MUST(cudaGetLastError());
    CUDA_MUST(cudaStreamSynchronize(cuStr.Get()));
  }
  _answers.emplace_back(_activeQuestion, iAnswer);
  _activeQuestion = cInvalidPqaId;
  return PqaError();
}

//// Instantiations
template class CudaQuiz<float>;

} // namespace ProbQA
