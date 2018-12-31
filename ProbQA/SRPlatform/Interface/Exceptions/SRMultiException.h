// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRSpinSync.h"

namespace SRPlat {

// Public methods of this class must be thread-safe.
class SRPLATFORM_API SRMultiException : public SRException {
  typedef SRPlat::SRSpinSync<16> TSync;
  class Impl;
  // This not only allows usage across modules (DLL/EXE), but also allows efficient storage of SRMultiException object
  //   without any std::unique_ptr etc. for the normal case when there are no exceptions in it.
  // Can't make it a std::unique_ptr, because the latter is not DLL-exported.
  Impl *_pImpl = nullptr;
  mutable TSync _sync;

private: // methods
  // This method is not thread-safe.
  Impl *DupImpl() const;
  // This method is not thread-safe.
  Impl *EnsureImpl();

  static SRString GetDefaultMessage() { return SRString::MakeUnowned("Multiple exceptions have occured."); }

public:
  virtual ~SRMultiException() override final;
  explicit SRMultiException() : SRException(GetDefaultMessage()) { }
  SRMultiException(const SRMultiException &fellow);
  SRMultiException& operator=(const SRMultiException &fellow);
  SRMultiException(SRMultiException &&fellow);
  SRMultiException& operator=(SRMultiException &&fellow);

  SREXCEPTION_TYPICAL(SRMulti);

  virtual SRString ToString() const override final;

  size_t AddException(SRException &&ex);
  // Take ownership of pEx and add it to the collection.
  size_t AddExceptionOwn(SRException *pEx);

  bool HasExceptions() const;
};

} // namespace SRPlat
