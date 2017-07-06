// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

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
    CalcTargetPriorsCache = 3,
    CalcTargetPriorsNocache = 4
  };

public: // variables
  CETask<taNumber> *_pTask;

public:
  virtual ~CESubtask() { }
  virtual Kind GetKind() = 0;
};

} // namespace ProbQA
