// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CECreateQuizOperation.decl.h"

namespace ProbQA {

inline CECreateQuizOpBase::CECreateQuizOpBase(PqaError& err) : _err(err) { }
inline CECreateQuizOpBase::~CECreateQuizOpBase() { };

inline CECreateQuizStart::CECreateQuizStart(PqaError& err) : CECreateQuizOpBase(err) { }
inline CECreateQuizOpBase::Operation CECreateQuizStart::GetCode() { return Operation::Start; }

template<typename taNumber> inline CECreateQuizResume<taNumber>::CECreateQuizResume(
  PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs)
  : CECreateQuizOpBase(err), _nAnswered(nAnswered), _pAQs(pAQs)
{ }

template<typename taNumber> inline CECreateQuizOpBase::Operation CECreateQuizResume<taNumber>::GetCode()
{ return Operation::Resume; }

} // namespace ProbQA
