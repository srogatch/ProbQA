// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEEvalQsTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CEEvalQsSubtaskConsider : public SRPlat::SRStandardSubtask {
public: // types
  typedef CEEvalQsTask<taNumber> TTask;

public: // constants
  static constexpr double _cMaxV = 2; //SRMath::_cSqrt2;
  static constexpr double _cLnMaxV = 0.69314718055994530941723212145818; // ln(2) //SRMath::_cLnSqrt2;
  static constexpr double _cLn0Stab = -746; // stabilizer for std::log(0)

private: // methods
  static double CalcVelocityComponent(const double V, const TPqaId nTargets);

public: // methods
  static size_t CalcStackReq(const EngineDefinition& engDef);

  using SRPlat::SRStandardSubtask::SRStandardSubtask;
  virtual void Run() override final;
};

} // namespace ProbQA
