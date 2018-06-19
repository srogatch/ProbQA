#include "stdafx.h"
#include "../PqaCore/CudaQuiz.h"
#include "../PqaCore/CudaEngine.h"
#include "../PqaCore/Interface/PqaErrorParams.h"

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
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "CUDA quiz is being implemented.")));
}

//// Instantiations
template class CudaQuiz<float>;

} // namespace ProbQA
