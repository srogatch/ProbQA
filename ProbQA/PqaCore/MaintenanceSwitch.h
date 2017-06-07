#pragma once

namespace ProbQA {

class MaintenanceSwitch {
public: // types
  enum class Mode : uint8_t {
    None = 0,
    Regular = 1,
    Maintenance = 2,
    Shutdown = 3
  };

  class AgnosticLock {
    MaintenanceSwitch *_pMs;
    Mode _mode;
  public:
    explicit AgnosticLock(MaintenanceSwitch& ms) : _pMs(&ms) {
      _mode = _pMs->EnterAgnostic();
    }
    ~AgnosticLock() {
      if (_pMs != nullptr) {
        _pMs->LeaveAgnostic();
      }
    }
    void EarlyRelease() {
      _pMs->LeaveAgnostic();
      _pMs = nullptr;
    }
    Mode GetMode() const {
      return _mode;
    }
  };

  // Object must be instantiated only after TryEnterSpecific() succeeds.
  template <Mode taMode> class SpecificLeaver {
    MaintenanceSwitch *_pMs;
  public:
    explicit SpecificLeaver(MaintenanceSwitch& ms) : _pMs(&ms) { }
    void EarlyRelease() {
      _pMs->LeaveSpecific<taMode>();
      _pMs = nullptr;
    }
    ~SpecificLeaver() {
      if (_pMs != nullptr) {
        _pMs->LeaveSpecific<taMode>();
      }
    }
  };

private: // variables
  SRPlat::SRCriticalSection _cs;
  SRPlat::SRConditionVariable _canSwitch;
  SRPlat::SRConditionVariable _canEnter;
  uint32_t _nUsing; // number of users of the current state - regular or maintenance
  uint32_t _curMode : 2; // current mode - regular or maintenance
  uint32_t _bModeChangeRequested : 1; // must be left |true| after shutdown
  uint32_t _bShutdownRequested : 1;

public: // methods
  static uint8_t ToUInt8(const Mode mode) { return static_cast<uint8_t>(mode); }

  explicit MaintenanceSwitch(Mode initMode);
  // Try to acquire the lock for a regular/maintenance-only operation, which delays the opposite mode until finished.
  // If the opposite mode is in progress, this method fails returning |false|.
  // NOTE: it dowsn't throw even when shut(ting) down: it returns |false| in this case.
  template <Mode taMode> bool TryEnterSpecific();
  template <Mode taMode> void LeaveSpecific();
  // Try to acquire the lock for an operation that can be run both in maintenance and regular mode.
  // Returns the mode, in which the lock has been obtained. In case this method is called during switching between
  //   regular and maintenance modes, it waits till the new mode is in effect.
  // Throws when shut(ting) down.
  Mode EnterAgnostic();
  void LeaveAgnostic();

  // Request to switch to another (maintenance/regular) mode: deny new operations of the current mode and wait for
  //   current operations of the current mode to finish, then switch to the new mode.
  // Throws if it is already in the target mode or in the process of state change.
  // Throws when shut(ting) down.
  template <Mode taMode> void SwitchMode();
  // Wait for completion of any current operations (except mode switch) and forbid starting any new operations in any
  //   mode. Concurrent switch operations throw and may do this a little later than when this method returns.
  // Returns |true| if shutdown has happened in the current call. Returns |false| if it was already shut down.
  bool Shutdown();
};

} // namespace ProbQA
