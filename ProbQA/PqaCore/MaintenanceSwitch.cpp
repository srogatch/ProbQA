#include "stdafx.h"
#include "../PqaCore/MaintenanceSwitch.h"

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

template <MaintenanceSwitch::Mode taMode> void MaintenanceSwitch::LeaveSpecific() {
  SRLock<TSync> sl(_sync);
  assert(static_cast<Mode>(_curMode) == taMode);
  assert(_nUsing >= 1);
  if ((--_nUsing) == 0 && _bModeChangeRequested) {
    //TODO: notify of the possibility to change state now.
  }
}

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

bool MaintenanceSwitch::EnableMaintenance() {

}
bool MaintenanceSwitch::DisableMaintenance() {

}

} // namespace ProbQA