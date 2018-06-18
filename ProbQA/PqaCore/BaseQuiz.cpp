// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/BaseQuiz.h"

using namespace SRPlat;

namespace ProbQA {

BaseQuiz::BaseQuiz(BaseEngine *pEngine) : _pEngine(pEngine) {
}

BaseQuiz::~BaseQuiz() {
}

} // namespace ProbQA
