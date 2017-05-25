#pragma once

#include "../PqaCore/Interface/PqaErrors.h"

namespace ProbQA {

class PQACORE_API NotImplementedErrorParams : public IPqaErrorParams {
  std::string _feature;
public:
  NotImplementedErrorParams(const std::string& feature) : _feature(feature) { }
  const std::string& GetFeature() { return _feature; }
};

} // namespace ProbQA
