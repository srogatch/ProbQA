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

template <MaintenanceSwitch::Mode taMode> void MaintenanceSwitch::SwitchMode() {
  {
    SRLock<SRCriticalSection> csl(_cs);
    if (_bModeChangeRequested) {
      if (_bShutdownRequested) {
        csl.EarlyRelease();
        throw PqaException(PqaErrorCode::ObjectShutDown, new ObjectShutDownErrorParams(SRString::MakeUnowned(
          __FUNCTION__ " at enter")));
      }
      else {
        uint8_t activeMode = ToUInt8(static_cast<Mode>(_curMode));
        csl.EarlyRelease();
        throw PqaException(PqaErrorCode::MaintenanceModeChangeInProgress, new MaintenanceModeErrorParams(activeMode));
      }
    }
    if (static_cast<Mode>(_curMode) == taMode) {
      csl.EarlyRelease();
      throw PqaException(PqaErrorCode::MaintenanceModeAlreadyThis, new MaintenanceModeErrorParams(ToUInt8(taMode)));
    }
    if (_nUsing > 0) {
      _bModeChangeRequested = 1;
      do {
        _canSwitch.Wait(_cs);
        if (_bShutdownRequested) {
          csl.EarlyRelease();
          throw PqaException(PqaErrorCode::ObjectShutDown, new ObjectShutDownErrorParams(SRString::MakeUnowned(
            __FUNCTION__ " at wait")));
        }
      } while (_nUsing > 0);
    }
    _curMode = static_cast<uint64_t>(taMode);
    _bModeChangeRequested = 0;
  }
  _canEnter.WakeAll();
}

template void MaintenanceSwitch::SwitchMode<MaintenanceSwitch::Mode::Maintenance>();
template void MaintenanceSwitch::SwitchMode<MaintenanceSwitch::Mode::Regular>();

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