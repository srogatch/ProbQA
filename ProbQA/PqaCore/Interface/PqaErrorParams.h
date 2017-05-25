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

} // namespace ProbQA
