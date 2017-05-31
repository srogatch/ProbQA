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
  static uint8_t ToUInt8(const Mode mode) { return static_cast<uint8_t>(mode); }

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

  // Request to switch to another (maintenance/regular) mode: deny new operations of the current mode and wait for
  //   current operations of the current mode to finish, then switch to the new mode.
  // Throws if it is already in the target mode or in the process of state change.
  template <Mode taMode> void SwitchMode();
};

} // namespace ProbQA
