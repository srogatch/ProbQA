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

} // namespace ProbQA
