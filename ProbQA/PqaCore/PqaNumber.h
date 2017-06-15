#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

class PqaNumber {
protected:
  PqaNumber() { }

public:
  PqaNumber(TPqaAmount init);
  PqaNumber& Mul(const PqaNumber& fellow);
  PqaNumber& Add(const PqaNumber& fellow);
  PqaNumber& operator+=(const TPqaAmount amount);
};

} // namespace ProbQA
