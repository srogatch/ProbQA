// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseCudaEngine.h"
#include "../PqaCore/Interface/CudaMain.h"
#include "../PqaCore/CudaMacros.h"

using namespace SRPlat;

namespace ProbQA {

BaseCudaEngine::BaseCudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi) : BaseEngine(engDef, pKbFi),
  _iDevice(0), _cspNb(false), _cuMp(256 * 1024 * 1024)
{
  CudaMain::Initialize(_iDevice);
  CUDA_MUST(cudaGetDeviceProperties(&_klc._cdp, _iDevice));
  _klc._maxBlocks = int32_t((int64_t(_klc._cdp.multiProcessorCount) * _klc._cdp.maxThreadsPerMultiProcessor)
    >> _klc._cLogBlockSize);
  if (_klc._cdp.warpSize != KernelLaunchContext::_cWarpSize) {
    SRException(SRMessageBuilder(SR_FILE_LINE "Unexpected CUDA warp size: ")(_klc._cdp.warpSize).GetOwnedSRString())
      .ThrowMoving();
  }
}

uint8_t* BaseCudaEngine::DevQuestionGaps() {
  return _gaps.Get();
}

uint8_t* BaseCudaEngine::DevTargetGaps() {
  return _gaps.Get() + ((_dims._nQuestions + 7) >> 3);
}

void BaseCudaEngine::BceUpdateWithDimensions(cudaStream_t stream) {
  _cuMp.FreeAllChunks();
  _gaps = CudaArray<uint8_t>(((_dims._nQuestions + 7) >> 3) + ((_dims._nTargets + 7) >> 3));
  CUDA_MUST(cudaMemcpyAsync(DevQuestionGaps(), _questionGaps.GetBits(), (_dims._nQuestions +7)>>3,
    cudaMemcpyHostToDevice, stream));
  CUDA_MUST(cudaMemcpyAsync(DevTargetGaps(), _targetGaps.GetBits(), (_dims._nTargets + 7) >> 3,
    cudaMemcpyHostToDevice, stream));
}

} // namespace ProbQA
