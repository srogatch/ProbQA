// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCuda/Interface/CudaMain.h"
#include "../PqaCuda/Utils.h"

namespace PqaCuda {

void CudaMain::SetDevice(const int iDevice, const bool bFirstInProcess) {
  CUDA_MUST(cudaSetDevice(iDevice));
  if (bFirstInProcess) {
    CUDA_MUST(cudaSetDeviceFlags(
      cudaDeviceScheduleYield // cudaDeviceScheduleBlockingSync //DEBUG
      | cudaDeviceMapHost
      //TODO: benchmark, not sure about this.
      // https://devtalk.nvidia.com/default/topic/621170/random-execution-times-and-freezes-with-concurent-kernels/
      | cudaDeviceLmemResizeToMax
    ));
  }
}

} // namespace PqaCuda
