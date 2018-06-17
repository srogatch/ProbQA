// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#ifdef PQACUDA_EXPORTS
#define PQACUDA_API __declspec(dllexport)
#else
#define PQACUDA_API __declspec(dllimport)
#endif
