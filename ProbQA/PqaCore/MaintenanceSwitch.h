#pragma once

namespace ProbQA {

class MaintenanceSwitch {
public: // types
  enum class State : uint8_t {
    None = 0,
    Regular = 1,
    Maintenance = 2
  };
  typedef SRPlat::SRSpinSync<32> TSync;

private: // variables
  std::atomic<int32_t> _nUsingRegular;
  std::atomic<int32_t> _nUsingMaintenance;
  TSync _sync;

public: // methods
  // Try to acquire the lock for a regular-only operation, which delays maintenance until finished.
  // If maintenance is in progress, this method fails returning |false|
  bool TryEnterRegular();
  void LeaveRegular();
  // Try to acquire the lock for an operation that can be run both in maintenance and regular mode.
  // Returns the mode, in which the lock has been obtained. In case this method is called during switching between
  //   regular and maintenance modes, it waits till the new mode is in effect.
  State EnterAgnostic();
  void LeaveAgnostic();
  // Try to acquire the lock for a maintenance-only operation.
  // If no maintenance is in progress, this method fails returning |false|
  bool TryEnterMaintenance();
  void LeaveMaintenance();

  // Request to start maintenance: deny new regular operations and wait for current regular operations to finish.
  void EnableMaintenance();
  // Request to stop maintenance: deny new maintenance operations and wait for the current maintenance operations to
  //   finish.
  void DisableMaintenance();
};

} // namespace ProbQA