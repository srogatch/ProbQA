// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/MaintenanceSwitch.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

MaintenanceSwitch::MaintenanceSwitch(Mode initMode) : _curMode(static_cast<uint64_t>(initMode)), 
  _nUsing(0), _bModeChangeRequested(0), _bShutdownRequested(0)
{

}

template <MaintenanceSwitch::Mode taMode> bool MaintenanceSwitch::TryEnterSpecific() {
  SRLock<SRCriticalSection> csl(_cs);
  if (static_cast<Mode>(_curMode) == taMode && !_bModeChangeRequested) {
    _nUsing++;
    return true;
  }
  else {
    return false;
  }
}

template bool MaintenanceSwitch::TryEnterSpecific<MaintenanceSwitch::Mode::Maintenance>();
template bool MaintenanceSwitch::TryEnterSpecific<MaintenanceSwitch::Mode::Regular>();

template <MaintenanceSwitch::Mode taMode> void MaintenanceSwitch::LeaveSpecific() {
  bool bWake = false;
  {
    SRLock<SRCriticalSection> csl(_cs);
    assert(static_cast<Mode>(_curMode) == taMode);
    assert(_nUsing >= 1);
    if ((--_nUsing) == 0 && _bModeChangeRequested) {
      // Notify of the possibility to switch mode now.
      bWake = true;
    }
  }
  if (bWake) {
    _canSwitch.WakeAll(); // switch and shutdown may be waiting simultaneously
  }
}

template void MaintenanceSwitch::LeaveSpecific<MaintenanceSwitch::Mode::Maintenance>();
template void MaintenanceSwitch::LeaveSpecific<MaintenanceSwitch::Mode::Regular>();

MaintenanceSwitch::Mode MaintenanceSwitch::EnterAgnostic() {
  SRLock<SRCriticalSection> csl(_cs);
  // Wait till mode switch request is fullfilled.
  while(_bModeChangeRequested) {
    if (_bShutdownRequested) {
      csl.EarlyRelease();
      throw PqaException(PqaErrorCode::ObjectShutDown, new ObjectShutDownErrorParams(SRString::MakeUnowned(
        __FUNCTION__ " at enter")));
    }
    _canEnter.Wait(_cs);
  }
  _nUsing++;
  return static_cast<Mode>(_curMode);
}

void MaintenanceSwitch::LeaveAgnostic() {
  bool bWake = false;
  {
    SRLock<SRCriticalSection> csl(_cs);
    assert(_nUsing >= 1);
    if ((--_nUsing) == 0 && _bModeChangeRequested) {
      // Notify of the possibility to switch mode now.
      bWake = true;
    }
  }
  if (bWake) {
    _canSwitch.WakeAll(); // switch and shutdown may be waiting simultaneously
  }
}

bool MaintenanceSwitch::Shutdown() {
  SRLock<SRCriticalSection> csl(_cs);
  if (_bShutdownRequested) {
    return false;
  }
  _bModeChangeRequested = 1;
  _bShutdownRequested = 1;
  while (_nUsing > 0) {
    _canSwitch.Wait(_cs);
  }
  return true;
}

} // namespace ProbQA
