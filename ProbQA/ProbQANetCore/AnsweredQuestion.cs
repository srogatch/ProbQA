using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  [StructLayout(LayoutKind.Sequential, Pack = 8)]
  public struct AnsweredQuestion
  {
    public Int64 _iQuestion;
    public Int64 _iAnswer;
  }
}
