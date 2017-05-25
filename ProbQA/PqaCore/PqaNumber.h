#pragma once

class PqaNumber {
protected:
  PqaNumber() { }

public:
  PqaNumber& Mul(const PqaNumber& fellow);
  PqaNumber& Add(const PqaNumber& fellow);
};