#include "stdafx.h"
#include "../PqaCore/MaintenanceSwitch.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

MaintenanceSwitch::MaintenanceSwitch(Mode initMode)
  : _curMode(static_cast<uint64_t>(initMode)), _nUsing(0), _bModeChangeRequested(false)
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
    _canSwitch.WakeAll();
  }
}

template void MaintenanceSwitch::LeaveSpecific<MaintenanceSwitch::Mode::Maintenance>();
template void MaintenanceSwitch::LeaveSpecific<MaintenanceSwitch::Mode::Regular>();

MaintenanceSwitch::Mode MaintenanceSwitch::EnterAgnostic() {
  SRLock<SRCriticalSection> csl(_cs);
  // Wait till mode switch request is fullfilled.
  while (_bModeChangeRequested) {
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
    _canSwitch.WakeAll();
  }
}

template <MaintenanceSwitch::Mode taMode> void MaintenanceSwitch::SwitchMode() {
  {
    SRLock<SRCriticalSection> csl(_cs);
    if (_bModeChangeRequested) {
      uint8_t activeMode = ToUInt8(static_cast<Mode>(_curMode));
      csl.EarlyRelease();
      throw PqaException(PqaErrorCode::MaintenanceModeChangeInProgress, new MaintenanceModeErrorParams(activeMode));
    }
    if (static_cast<Mode>(_curMode) == taMode) {
      csl.EarlyRelease();
      throw PqaException(PqaErrorCode::MaintenanceModeAlreadyThis, new MaintenanceModeErrorParams(ToUInt8(taMode)));
    }
    if (_nUsing > 0) {
      _bModeChangeRequested = 1;
      while (_nUsing > 0) {
        _canSwitch.Wait(_cs);
      }
    }
    _curMode = static_cast<uint64_t>(taMode);
    _bModeChangeRequested = 0;
  }
  _canEnter.WakeAll();
}

template void MaintenanceSwitch::SwitchMode<MaintenanceSwitch::Mode::Maintenance>();
template void MaintenanceSwitch::SwitchMode<MaintenanceSwitch::Mode::Regular>();

} // namespace ProbQA