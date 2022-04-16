// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

ATTR_RESTRICT char* SRString::DupData(const char *PTR_RESTRICT const pData, const size_t length) {
  //TODO: shall here rather be aligned memory allocation?
  char *PTR_RESTRICT const pDup = new char[length];
  memcpy(pDup, pData, length);
  return pDup;
}

void SRString::setSameData(const char *PTR_RESTRICT const pData) {
  if (_bOwned) {
    _pData = DupData(pData, _length);
  }
  else {
    _pData = pData;
  }
}

SRString::SRString() : _pData(nullptr), _bOwned(0), _length(0) {
}

SRString::SRString(const char *PTR_RESTRICT const pData, const bool bOwned, const size_t length)
  : _pData(pData), _bOwned(bOwned ? 1 : 0) {
  if (length == std::string::npos) {
    _length = strlen(pData);
  }
  else {
    _length = length;
  }
}

SRString::SRString(const SRString& fellow) {
  _length = fellow._length;
  _bOwned = fellow._bOwned;
  setSameData(fellow._pData);
}

SRString& SRString::operator=(const SRString& fellow) {
  if (this != &fellow) {
    if (_bOwned) {
      delete _pData;
    }
    _length = fellow._length;
    _bOwned = fellow._bOwned;
    setSameData(fellow._pData);
  }
  return *this;
}

SRString::SRString(SRString&& fellow) noexcept : _pData(fellow._pData), _bOwned(fellow._bOwned), _length(fellow._length) {
  fellow._pData = nullptr;
  fellow._bOwned = 0;
  fellow._length = 0;
}

SRString& SRString::operator=(SRString&& fellow) {
  if (this != &fellow) {
    if (_bOwned) {
      delete _pData;
    }
    this->_pData = fellow._pData;
    this->_bOwned = fellow._bOwned;
    this->_length = fellow._length;
    fellow._pData = nullptr;
    fellow._bOwned = 0;
    fellow._length = 0;
  }
  return *this;
}

SRString::~SRString() {
  if (_bOwned) {
    delete _pData;
  }
}

SRString::SRString(const std::string& source) : SRString(DupData(source.c_str(), source.size()), true, source.size()) {
}

SRString SRString::MakeOwned(const char *PTR_RESTRICT const pData, size_t length) {
  return SRString(pData, true, length);
}

SRString SRString::MakeClone(const char *PTR_RESTRICT const pData, size_t length) {
  if (length == std::string::npos) {
    length = strlen(pData);
  }
  return SRString(DupData(pData, length), true, length);
}

SRString SRString::MakeUnowned(const char *PTR_RESTRICT const pData, size_t length) {
  return SRString(pData, false, length);
}

size_t SRString::GetData(const char *PTR_RESTRICT &PTR_RESTRICT outData) const {
  outData = _pData;
  return _length;
}

} // namespace SRPlat
