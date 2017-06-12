#include "stdafx.h"
#include "../PqaCore/Interface/PqaErrorParams.h"

using namespace SRPlat;

namespace ProbQA {

////////////////////////////////  AggregateErrorParams /////////////////////////////////////////////////////////////////

class AggregateErrorParams::Impl {
public:
  std::vector<PqaError> _errors;
  Impl() { }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
};

AggregateErrorParams::AggregateErrorParams() {
  _pImpl = new Impl();
}
AggregateErrorParams::~AggregateErrorParams() {
  delete _pImpl;
}

void AggregateErrorParams::Add(PqaError&& pe) {
  _pImpl->_errors.push_back(std::forward<PqaError>(pe));
}

SRPlat::SRString AggregateErrorParams::ToString() {
  std::vector<PqaError>& errImpl = _pImpl->_errors;
  SRMessageBuilder mb("Aggregate error");
  for (size_t i = 0, iEn=errImpl.size(); i < iEn; i++) {
    mb(" [");
    mb(errImpl[i].ToString(true));
    mb.AppendChar(']');
  }
  return mb.GetOwnedSRString();
}

size_t AggregateErrorParams::Count() {
  return _pImpl->_errors.size();
}

} // namespace ProbQA
