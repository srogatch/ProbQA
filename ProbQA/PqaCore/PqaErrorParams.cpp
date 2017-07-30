// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

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

AggregateErrorParams::AggregateErrorParams(Impl *pImpl) : _pImpl(pImpl) {
}

AggregateErrorParams::AggregateErrorParams() : _pImpl(nullptr) {
}

AggregateErrorParams::~AggregateErrorParams() {
  delete _pImpl; // if it's not nullptr
}

AggregateErrorParams::Impl* AggregateErrorParams::EnsureImpl() {
  if (_pImpl == nullptr) {
    _pImpl = new Impl();
  }
  return _pImpl;
}

void AggregateErrorParams::Add(PqaError&& pe) {
  EnsureImpl()->_errors.push_back(std::forward<PqaError>(pe));
}

SRPlat::SRString AggregateErrorParams::ToString() {
  if (_pImpl == nullptr) {
    return SRString::MakeUnowned("Success (empty aggregate error).");
  }
  std::vector<PqaError>& errImpl = _pImpl->_errors;
  SRMessageBuilder mb("Aggregate error");
  for (size_t i = 0, iEn=errImpl.size(); i < iEn; i++) {
    mb(" [");
    mb(errImpl[i].ToString(true));
    mb.AppendChar(']');
  }
  return mb.GetOwnedSRString();
}

size_t AggregateErrorParams::Count() const {
  if (_pImpl == nullptr) {
    return 0;
  }
  return _pImpl->_errors.size();
}

AggregateErrorParams* AggregateErrorParams::Move() {
  AggregateErrorParams *ans = new AggregateErrorParams(_pImpl);
  _pImpl = nullptr;
  return ans;
}

} // namespace ProbQA
