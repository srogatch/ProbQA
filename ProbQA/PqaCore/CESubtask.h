#pragma once

namespace ProbQA {

template <typename taNumber> class CETask;

template<typename taNumber> class CESubtask {
public: // types
  // These must be sequential because engine's pool has a vector of subtask chains
  enum class Kind : uint8_t {
    None = 0,
    TrainDistrib = 1,
    TrainAdd = 2,
    CalcTargetPriors = 3
  };

public: // variables
  CETask<taNumber> *_pTask;

public:
  virtual ~CESubtask() { }
  virtual Kind GetKind() = 0;
};

} // namespace ProbQA
