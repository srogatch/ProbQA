using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  public class PqaEngineFactory
  {
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    private class CiEngineDefinition
    {
      public long _nAnswers;
      public long _nQuestions;
      public long _nTargets;
      public byte _precType;
      public ushort _precExponent;
      public uint _precMantissa;
      public double _initAmount;
      public long _memPoolMaxBytes;
    }

    [DllImport("PqaCore.dll")]
    private static extern IntPtr CiPqaGetEngineFactory();

    [DllImport("PqaCore.dll")]
    private static extern IntPtr CiPqaEngineFactory_CreateCpuEngine(out IntPtr pError, CiEngineDefinition ciEngDef);

    private static PqaEngineFactory _instance;
    private static Object _sync = new Object();

    private IntPtr _nativeFactory;

    private PqaEngineFactory(IntPtr nativeFactory)
    {
      _nativeFactory = nativeFactory;
    }

    public static PqaEngineFactory Instance()
    {
      if (_instance != null)
      {
        return _instance;
      }
      lock (_sync)
      {
        if (_instance != null)
        {
          return _instance;
        }
        _instance = new PqaEngineFactory(CiPqaGetEngineFactory());
      }
      return _instance;
    }

    PqaEngine CreateCpuEngine(out PqaError err, EngineDefinition engDef)
    {
      if(engDef.AnswerCount == null || engDef.QuestionCount == null || engDef.TargetCount == null)
      {
        throw new Exception("Answer, question and target counts must be all set.");
      }
      IntPtr nativeError = IntPtr.Zero;
      CiEngineDefinition ciEngDef = new CiEngineDefinition() {
        _nAnswers = engDef.AnswerCount.Value,
        _nQuestions = engDef.QuestionCount.Value,
        _nTargets = engDef.TargetCount.Value,
        _precType = (byte)engDef.PrecType,
        _precExponent = engDef.PrecExponent,
        _precMantissa = engDef.PrecMantissa,
        _initAmount = engDef.InitAmount,
        _memPoolMaxBytes = engDef.MemPoolMaxBytes
      };
      IntPtr nativeEngine = CiPqaEngineFactory_CreateCpuEngine(out nativeError, ciEngDef);
      err = new PqaError(nativeError);
      return new PqaEngine(nativeEngine);
    }
  }
}
