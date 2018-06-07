using System;
using System.Collections.Generic;
using System.Text;

namespace ProbQANetCore
{
  public class EngineDefinition
  {
    public const ulong cDefaultMemPoolMaxBytes = 512 * 1024 * 1024;

    public long? AnswerCount { get; set; }
    public long? QuestionCount { get; set; }
    public long? TargetCount { get; set; }
    public PrecisionType PrecType { get; set; } = PrecisionType.Double;
    public ushort PrecExponent { get; set; } = 11;
    public uint PrecMantissa { get; set; } = 53;
    public double InitAmount { get; set; } = 1.0;
    public ulong MemPoolMaxBytes { get; set; } = cDefaultMemPoolMaxBytes;
  }
}
