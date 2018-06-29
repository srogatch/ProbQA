// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/Interface/CudaMain.h"
#include "../PqaCore/CudaMacros.h"

using namespace SRPlat;

namespace ProbQA {

struct CudaDeviceLockInfo {
  typedef SRSpinSync<(1 << 8)> TSync;
  int _iDevice;
  int64_t _nRefs;
  TSync _sync;

  void ChangeRefs(const int8_t delta) {
    bool noRefsLeft;
    {
      SRLock<TSync> sl(_sync);
      _nRefs += delta;
      assert(_nRefs >= 0);
      noRefsLeft = (_nRefs == 0);
    }
    if (noRefsLeft) {
      CudaMain::_cvCanSwitchDevice.WakeAll();
    }
  }
};

namespace {
  CudaDeviceLockInfo gCdli;
  std::unordered_set<int> gDevsInitialized;
}

/////////////////////// CudaDeviceLock class implementation ////////////////////////////////////////////////////////////
CudaDeviceLock::CudaDeviceLock(const CudaDeviceLock &fellow) {
  if (!fellow._bAcquired) {
    return;
  }
  SRLock<CudaDeviceLockInfo::TSync> sl(gCdli._sync);
  gCdli._nRefs++;
}

CudaDeviceLock& CudaDeviceLock::operator=(const CudaDeviceLock &fellow) {
  const int8_t refChange = (fellow._bAcquired ? 1 : 0) - (_bAcquired ? 1 : 0);
  if (refChange == 0) { // covers the case &fellow==this
    return *this;
  }
  assert(&fellow != this);
  gCdli.ChangeRefs(refChange);
  return *this;
}

CudaDeviceLock::CudaDeviceLock(CudaDeviceLock &&fellow) {
  _bAcquired = fellow._bAcquired;
  fellow._bAcquired = false;
}

CudaDeviceLock& CudaDeviceLock::operator=(CudaDeviceLock &&fellow) {
  if (this == &fellow) {
    return *this;
  }
  const bool wasAcquired = _bAcquired;
  _bAcquired = fellow._bAcquired;
  fellow._bAcquired = false;
  if (wasAcquired) {
    gCdli.ChangeRefs(-1);
  }
  return *this;
}

CudaDeviceLock::~CudaDeviceLock() {
  if (_bAcquired) {
    gCdli.ChangeRefs(-1);
  }
}

/////////////////////// CudaMain class members /////////////////////////////////////////////////////////////////////////

SRCriticalSection CudaMain::_csInit;
SRConditionVariable CudaMain::_cvCanSwitchDevice;
SRCriticalSection CudaMain::_csDeviceSwitch;

bool CudaMain::IsInitialized(const int iDevice) {
  SRLock<SRCriticalSection> csl(_csInit);
  return gDevsInitialized.find(iDevice) != gDevsInitialized.end();
}

bool CudaMain::Initialize(const int iDevice) {
  SRLock<SRCriticalSection> csl(_csInit);
  if (gDevsInitialized.find(iDevice) != gDevsInitialized.end()) {
    return false;
    //csl.EarlyRelease();
    //SRException(SRMessageBuilder(SR_FILE_LINE "CUDA device ")(iDevice)(" is already initialized.").GetOwnedSRString())
    //  .ThrowMoving();
  }
  CudaDeviceLock cdl = SetDevice(iDevice);

  CUDA_MUST(cudaSetDeviceFlags(
    //cudaDeviceScheduleBlockingSync
    cudaDeviceScheduleYield // cudaDeviceScheduleBlockingSync //DEBUG
    | cudaDeviceMapHost
    //TODO: benchmark, not sure about this.
    // https://devtalk.nvidia.com/default/topic/621170/random-execution-times-and-freezes-with-concurent-kernels/
    | cudaDeviceLmemResizeToMax
  ));
  gDevsInitialized.insert(iDevice);
  return true;
}

template<bool taTryOnly> CudaDeviceLock CudaMain::SetDeviceInternal(const int iDevice) {
  SRLock<SRCriticalSection> csl(_csDeviceSwitch);
  for (;;) {
    {
      SRLock<CudaDeviceLockInfo::TSync> sl(gCdli._sync);
      if (gCdli._nRefs == 0) {
        // We are holding the spin lock for a long time here, however, noone is expected to try to acquire it in the
        //  meantime because the reference counter is 0, so everyone has first to acquire critical section
        //  _csDeviceSwitch, which we are holding anyway
        CUDA_MUST(cudaSetDevice(iDevice));
        gCdli._nRefs = 1;
        gCdli._iDevice = iDevice;
        break;
      }
      if (gCdli._iDevice == iDevice) {
        gCdli._nRefs++;
        break;
      }
    }
    if constexpr(taTryOnly) {
      return CudaDeviceLock();
    }
    _cvCanSwitchDevice.Wait(_csDeviceSwitch);
  }
  return CudaDeviceLock(true);
}

CudaDeviceLock CudaMain::SetDevice(const int iDevice) {
  return SetDeviceInternal<false>(iDevice);
}

CudaDeviceLock CudaMain::TrySetDevice(const int iDevice) {
  return SetDeviceInternal<true>(iDevice);
}

void CudaMain::FlushWddm(cudaStream_t stream) {
  cudaError_t status = cudaStreamQuery(stream);
  if (status != cudaSuccess && status != cudaErrorNotReady) {
    ProbQA::CudaException(status, SRPlat::SRMessageBuilder(SR_FILE_LINE)("CUDA error #")(status)(": ") \
      (cudaGetErrorString(status)).GetOwnedSRString()).ThrowMoving();
  }
}

} // namespace ProbQA
