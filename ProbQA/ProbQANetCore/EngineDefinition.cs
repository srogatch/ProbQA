using System;
using System.Collections.Generic;
using System.Text;

namespace ProbQANetCore
{
  public class EngineDefinition
  {
    public long? AnswerCount { get; set; }
    public long? QuestionCount { get; set; }
    public long? TargetCount { get; set; }
    public PrecisionType PrecType { get; set; } = PrecisionType.Double;
    public ushort PrecExponent { get; set; } = 11;
    public uint PrecMantissa { get; set; } = 53;
    public double InitAmount { get; set; } = 1.0;
    public long MemPoolMaxBytes { get; set; } = 512 * 1024 * 1024;
  }
}
