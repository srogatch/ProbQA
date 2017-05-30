#pragma once

namespace ProbQA {

class MaintenanceSwitch {
public: // types
  enum class Mode : uint8_t {
    None = 0,
    Regular = 1,
    Maintenance = 2
  };
  typedef SRPlat::SRSpinSync<32> TSync;

private: // variables
  uint64_t _nUsing : 32; // number of users of the current state - regular or maintenance
  uint64_t _curMode : 2; // current mode - regular or maintenance
  uint64_t _bModeChangeRequested : 1;
  TSync _sync;

public: // methods
  explicit MaintenanceSwitch(Mode initMode);
  // Try to acquire the lock for a regular/maintenance-only operation, which delays the opposite mode until finished.
  // If the opposite mode is in progress, this method fails returning |false|.
  template <Mode taMode> bool TryEnterSpecific();
  template <Mode taMode> void LeaveSpecific();
  // Try to acquire the lock for an operation that can be run both in maintenance and regular mode.
  // Returns the mode, in which the lock has been obtained. In case this method is called during switching between
  //   regular and maintenance modes, it waits till the new mode is in effect.
  Mode EnterAgnostic();
  void LeaveAgnostic();

  // Request to start maintenance: deny new regular operations and wait for current regular operations to finish.
  // Returns |false| if it is already in maintenance.
  bool EnableMaintenance();
  // Request to stop maintenance: deny new maintenance operations and wait for the current maintenance operations to
  //   finish.
  // Returns |false| it is already out of maintenance.
  bool DisableMaintenance();
};

} // namespace ProbQA
