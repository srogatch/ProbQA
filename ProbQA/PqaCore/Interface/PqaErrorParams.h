#pragma once

#include "../PqaCore/Interface/PqaErrors.h"
#include "../SRPlatform/Interface/SRString.h"

namespace ProbQA {

class PQACORE_API NotImplementedErrorParams : public IPqaErrorParams {
  SRPlat::SRString _feature;
public:
  NotImplementedErrorParams(const SRPlat::SRString& feature) : _feature(feature) { }
  const SRPlat::SRString& GetFeature() { return _feature; }
};

class PQACORE_API CommonExceptionErrorParams : public IPqaErrorParams {
  SRPlat::SRString _etn; // exception type name
public:
  template <typename T> explicit CommonExceptionErrorParams(const T& ex) {
    _etn = SRPlat::SRString::MakeUnowned(typeid(ex).name());
  }
  const SRPlat::SRString& GetExceptionTypeName() { return _etn; }
};

class PQACORE_API InsufficientEngineDimensionsErrorParams : public IPqaErrorParams {
public: // variables
  const TPqaId _nAnswers;
  const TPqaId _minAnswers;
  const TPqaId _nQuestions;
  const TPqaId _minQuestions;
  const TPqaId _nTargets;
  const TPqaId _minTargets;
public: // methods
  InsufficientEngineDimensionsErrorParams(const TPqaId nAnswers, const TPqaId minAnswers,
    const TPqaId nQuestions, const TPqaId minQuestions, const TPqaId nTargets, const TPqaId minTargets)
    : _nAnswers(nAnswers), _minAnswers(minAnswers), _nQuestions(nQuestions), _minQuestions(minQuestions),
    _nTargets(nTargets), _minTargets(minTargets)
  {
  }
};

class PQACORE_API MaintenanceModeErrorParams : public IPqaErrorParams {
  uint8_t _activeMode;
public:
  explicit MaintenanceModeErrorParams(const uint8_t activeMode) : _activeMode(activeMode) { }
  uint8_t GetActiveMode() const { return _activeMode; }
};

} // namespace ProbQA
