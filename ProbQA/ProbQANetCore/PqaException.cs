using System;
using System.Collections.Generic;
using System.Text;

namespace ProbQANetCore
{
  public class PqaException : Exception
  {
    public PqaException(string message) : base(message)
    {
    }
  }
}
