// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

class SRPLATFORM_API SRDefaultLoggerAlreadyInitializedException : public SRException {
public:
  explicit SRDefaultLoggerAlreadyInitializedException(SRString &&message) 
    : SRException(std::forward<SRString>(message)) { }
};

class SRPLATFORM_API SRDefaultLoggerConcurrentInitializationException : public SRException {
public:
  explicit SRDefaultLoggerConcurrentInitializationException(SRString &&message)
    : SRException(std::forward<SRString>(message)) { }
};

} // namespace SRPlat
