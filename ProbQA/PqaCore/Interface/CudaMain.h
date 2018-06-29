// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

struct CudaDeviceLockInfo;
class CudaMain;

// Watch the order of this lock w.r.t. your other synchronization locks.
class PQACORE_API CudaDeviceLock {
  friend class CudaMain;

private: // variables
  bool _bAcquired;

private: // methods
  explicit CudaDeviceLock(const bool acquired) : _bAcquired(acquired) { }

public: // methods
  explicit CudaDeviceLock() : _bAcquired(false) { }
  CudaDeviceLock(const CudaDeviceLock &fellow);
  CudaDeviceLock& operator=(const CudaDeviceLock &fellow);
  CudaDeviceLock(CudaDeviceLock &&fellow);
  CudaDeviceLock& operator=(CudaDeviceLock &&fellow);
  ~CudaDeviceLock();

  bool IsAcquired() const { return _bAcquired; }
};

class PQACORE_API CudaMain {
  friend struct CudaDeviceLockInfo;

private: // variables
  static SRPlat::SRCriticalSection _csInit; // acquired before _csDeviceSwitch
  static SRPlat::SRCriticalSection _csDeviceSwitch;
  static SRPlat::SRConditionVariable _cvCanSwitchDevice;

private: // methods
  template<bool taTryOnly> static CudaDeviceLock SetDeviceInternal(const int iDevice);

public:
  // Initializes the device and temporarily sets it as current.
  static bool Initialize(const int iDevice);
  static bool IsInitialized(const int iDevice);
  static CudaDeviceLock SetDevice(const int iDevice);
  static CudaDeviceLock TrySetDevice(const int iDevice);
  static void FlushWddm(cudaStream_t stream);
};

} // namespace ProbQA
