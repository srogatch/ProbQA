// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/Exceptions/SRMultiException.h"
#include "../SRPlatform/Interface/SRLock.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

// This class is not intended thread-safe.
class SRMultiException::Impl {
  std::vector<std::unique_ptr<SRException>> _exPtrs;
public:
  Impl* Clone() const {
    std::unique_ptr<Impl> answer(new Impl());
    for (size_t i = 0; i < _exPtrs.size(); i++) {
      answer->_exPtrs.emplace_back(_exPtrs[i]->Clone());
    }
    return answer.release();
  }
  size_t AddException(SRException &&ex) {
    _exPtrs.emplace_back(ex.Move());
    return _exPtrs.size();
  }
  size_t AddExceptionOwn(SRException *pEx) {
    _exPtrs.emplace_back(pEx);
    return _exPtrs.size();
  }
  size_t GetCount() {
    return _exPtrs.size();
  }
  SRException& GetAt(const size_t idx) {
    return *(_exPtrs[idx]);
  }
};

///////////////////////////////////////// SRMultiException implementation //////////////////////////////////////////////

SRMultiException::Impl *SRMultiException::DupImpl() const {
  if (_pImpl == nullptr) {
    return nullptr;
  }
  else {
    return _pImpl->Clone();
  }
}

SRMultiException::~SRMultiException() {
  delete _pImpl;
}

SRMultiException::SRMultiException(const SRMultiException &fellow) : SRException(GetDefaultMessage()) {
  SRLock<TSync> slF(fellow._sync);
  _message = fellow._message;
  _pImpl = fellow.DupImpl();
}

SRMultiException& SRMultiException::operator=(const SRMultiException &fellow) {
  if (this != &fellow) {
    std::unique_ptr<Impl> pOldImpl;
    // Avoid deadlocks if both objects are used asynchronously, by ordering the locks by their addresses.
    SRLock<TSync> slL(this < &fellow ? _sync : fellow._sync);
    SRLock<TSync> slG(this < &fellow ? fellow._sync : _sync);
    std::unique_ptr<Impl> pDupImpl(fellow.DupImpl());
    _message = fellow._message;
    pOldImpl.reset(_pImpl);
    _pImpl = pDupImpl.release();
  }
  return *this;
}

SRMultiException::SRMultiException(SRMultiException &&fellow) noexcept : SRException(GetDefaultMessage()) {
  SRLock<TSync> fsl(fellow._sync);
  std::swap(_message, fellow._message);
  _pImpl = fellow._pImpl;
  fellow._pImpl = nullptr;
}

SRMultiException& SRMultiException::operator=(SRMultiException &&fellow) {
  if (this != &fellow) {
    std::unique_ptr<Impl> pOldImpl;
    SRString &&nullMessage = GetDefaultMessage();
    SRLock<TSync> slL(this < &fellow ? _sync : fellow._sync);
    SRLock<TSync> slG(this < &fellow ? fellow._sync : _sync);
    _message = std::forward<SRString>(fellow._message);
    fellow._message = std::move(nullMessage);
    pOldImpl.reset(_pImpl);
    _pImpl = fellow._pImpl;
    fellow._pImpl = nullptr;
  }
  return *this;
}

SRMultiException::Impl* SRMultiException::EnsureImpl() {
  if (_pImpl == nullptr) {
    _pImpl = new Impl();
  }
  return _pImpl;
}

size_t SRMultiException::AddException(SRException &&ex) {
  SRLock<TSync> sl(_sync);
  return EnsureImpl()->AddException(std::forward<SRException>(ex));
}

size_t SRMultiException::AddExceptionOwn(SRException *pEx) {
  SRLock<TSync> sl(_sync);
  return EnsureImpl()->AddExceptionOwn(pEx);
}

SRString SRMultiException::ToString() const {
  SRLock<TSync> sl(_sync);
  size_t nExs;
  if (_pImpl == nullptr || (nExs = _pImpl->GetCount()) == 0) {
    return SRString::MakeUnowned("No exceptions on record in the SRMultiException.");
  }
  SRMessageBuilder mb(_message);
  for (size_t i = 0; i < nExs; i++) {
    mb(" Exception #")(i)(": [")(_pImpl->GetAt(i).ToString())("];");
  }
  return mb.GetOwnedSRString();
}

bool SRMultiException::HasExceptions() const {
  SRLock<TSync> sl(_sync);
  return _pImpl != nullptr && _pImpl->GetCount() > 0;
}

} // namespace SRPlat
