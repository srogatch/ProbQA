// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRDefaultLogger.h"
#include "../SRPlatform/DbgLogger.h"
#include "../SRPlatform/Interface/Exceptions/SRDefaultLoggerExceptions.h"
#include "../SRPlatform/Interface/SRLoggerFactory.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

namespace {

class DefaultLoggerImpl {
  static std::atomic<DefaultLoggerImpl*> _pInst;
  std::atomic<ISRLogger*> _pFileLogger;
  std::unique_ptr<DbgLogger> _pDbgLogger;

protected: // Methods
  // Singleton created by Instance() method.
  explicit DefaultLoggerImpl();

public: // Methods
  static DefaultLoggerImpl* Instance();
  void Init(const SRString& baseName);
  ISRLogger* Get();
  ISRLogger* Dbg();
};

// Never destruct this because loggers may be needed till the very end of the program.
std::atomic<DefaultLoggerImpl*> DefaultLoggerImpl::_pInst(nullptr);

DefaultLoggerImpl* DefaultLoggerImpl::Instance() {
  DefaultLoggerImpl *pDli = _pInst.load(std::memory_order_acquire);
  if (pDli != nullptr) {
    return pDli;
  }
  DefaultLoggerImpl *pOther = nullptr;
  pDli = new DefaultLoggerImpl();
  if (_pInst.compare_exchange_strong(pOther, pDli, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return pDli;
  }
  delete pDli;
  return pOther;
}

DefaultLoggerImpl::DefaultLoggerImpl() : _pDbgLogger(new DbgLogger()), _pFileLogger(nullptr) {
}

void DefaultLoggerImpl::Init(const SRString& baseName) {
  if (_pFileLogger.load(std::memory_order_relaxed)) {
    throw SRDefaultLoggerAlreadyInitializedException(SRString::MakeUnowned("Default logger seems already initialized"
      " by the moment of calling DefaultLoggerImpl::Init()."));
  }
  ISRLogger *pFileLogger = SRLoggerFactory::MakeFileLogger(baseName);
  ISRLogger *pOther = nullptr;
  if (!_pFileLogger.compare_exchange_strong(pOther, pFileLogger, std::memory_order_release, std::memory_order_relaxed))
  {
    SRString message = SRMessageBuilder("Detected concurrent initialization of the default logger, with the previous"
      " file logger at address ")(static_cast<const void*>(pOther)).GetOwnedSRString();
    // This is doubtful, but I think if we created the file and then immediately abandon it, we should note why.
    pFileLogger->Log(ISRLogger::Severity::Error, message);
    delete pFileLogger;
    throw SRDefaultLoggerConcurrentInitializationException(std::move(message));
  }
}

ISRLogger* DefaultLoggerImpl::Get() {
  ISRLogger *pFileLogger = _pFileLogger.load(std::memory_order_acquire);
  if (pFileLogger != nullptr) {
    return pFileLogger;
  }
  return _pDbgLogger.get();
}

ISRLogger* DefaultLoggerImpl::Dbg() {
  return _pDbgLogger.get();
}

} // Anonymous namespace

void SRDefaultLogger::Init(const SRString& baseName) {
  return DefaultLoggerImpl::Instance()->Init(baseName);
}

ISRLogger* SRDefaultLogger::Get() {
  return DefaultLoggerImpl::Instance()->Get();
}

ISRLogger* SRDefaultLogger::Dbg() {
  return DefaultLoggerImpl::Instance()->Dbg();
}

} // namespace SRPlat
