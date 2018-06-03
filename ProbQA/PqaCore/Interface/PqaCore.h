// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

//NOTE: this file can be included from both C and C++

#ifdef PQACORE_EXPORTS
#define PQACORE_API __declspec(dllexport)
#else
#define PQACORE_API __declspec(dllimport)
#endif

