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
  SRLock<TSync> sl(_sync);
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
  SRLock<TSync> sl(_sync);
  assert(static_cast<Mode>(_curMode) == taMode);
  assert(_nUsing >= 1);
  if ((--_nUsing) == 0 && _bModeChangeRequested) {
    //TODO: notify of the possibility to change state now.
  }
}

template void MaintenanceSwitch::LeaveSpecific<MaintenanceSwitch::Mode::Maintenance>();
template void MaintenanceSwitch::LeaveSpecific<MaintenanceSwitch::Mode::Regular>();

MaintenanceSwitch::Mode MaintenanceSwitch::EnterAgnostic() {
  {
    SRLock<TSync> sl(_sync);
    if (!_bModeChangeRequested) {
      _nUsing++;
      return static_cast<Mode>(_curMode);
    }
  }
  //TODO: otherwise, wait till state change request is fullfilled.
}

void MaintenanceSwitch::LeaveAgnostic() {
  SRLock<TSync> sl(_sync);
  assert(_nUsing >= 1);
  if ((--_nUsing) == 0 && _bModeChangeRequested) {
    //TODO: notify of the possibility to change state now.
  }
}

template <MaintenanceSwitch::Mode taMode> void MaintenanceSwitch::SwitchMode() {
  {
    SRLock<TSync> sl(_sync);
    if (_bModeChangeRequested) {
      uint8_t activeMode = ToUInt8(static_cast<Mode>(_curMode));
      sl.EarlyRelease();
      throw PqaException(PqaErrorCode::MaintenanceModeChangeInProgress, new MaintenanceModeErrorParams(activeMode));
    }
    if (static_cast<Mode>(_curMode) == taMode) {
      sl.EarlyRelease();
      throw PqaException(PqaErrorCode::MaintenanceModeAlreadyThis, new MaintenanceModeErrorParams(ToUInt8(taMode)));
    }
    if (_nUsing == 0) {
      _curMode = static_cast<uint64_t>(taMode);
      return;
    }
    _bModeChangeRequested = 1;
    //TODO: reset the event while in the lock
  }
  //TODO: implement
}

template void MaintenanceSwitch::SwitchMode<MaintenanceSwitch::Mode::Maintenance>();
template void MaintenanceSwitch::SwitchMode<MaintenanceSwitch::Mode::Regular>();

} // namespace ProbQA