using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  public class PqaEngineFactory
  {
    // Cannot promote this to EngineDefinition because structs cannot have default initializers
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    private struct CiEngineDefinition
    {
      public Int64 _nAnswers;
      public Int64 _nQuestions;
      public Int64 _nTargets;
      public byte _precType;
      public UInt16 _precExponent;
      public UInt32 _precMantissa;
      public double _initAmount;
      public UInt64 _memPoolMaxBytes;
    }

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr CiGetPqaEngineFactory();

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr PqaEngineFactory_CreateCpuEngine(IntPtr pFactory, ref IntPtr ppError,
      ref CiEngineDefinition pEngDef);

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr PqaEngineFactory_LoadCpuEngine(IntPtr pFactory, ref IntPtr ppError, string filePath,
      UInt64 memPoolMaxBytes = EngineDefinition.cDefaultMemPoolMaxBytes);

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
        _instance = new PqaEngineFactory(CiGetPqaEngineFactory());
      }
      return _instance;
    }

    public PqaEngine CreateCpuEngine(out PqaError err, EngineDefinition engDef)
    {
      if(engDef.AnswerCount == null || engDef.QuestionCount == null || engDef.TargetCount == null)
      {
        throw new PqaException("Answer, question and target counts must be all set.");
      }
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
      IntPtr nativeEngine;
      IntPtr nativeError = IntPtr.Zero;
      try
      {
        nativeEngine = PqaEngineFactory_CreateCpuEngine(_nativeFactory, ref nativeError, ref ciEngDef);
      }
      finally
      {
        err = PqaError.Factor(nativeError);
      }
      if(nativeEngine == IntPtr.Zero)
      {
        return null;
      }
      return new PqaEngine(nativeEngine);
    }

    public PqaEngine LoadCpuEngine(out PqaError err, string filePath,
      ulong memPoolMaxBytes = EngineDefinition.cDefaultMemPoolMaxBytes)
    {
      IntPtr nativeEngine;
      IntPtr nativeError = IntPtr.Zero;
      try
      {
        nativeEngine = PqaEngineFactory_LoadCpuEngine(_nativeFactory, ref nativeError, filePath, memPoolMaxBytes);
      }
      finally
      {
        err = PqaError.Factor(nativeError);
      }
      if(nativeEngine == IntPtr.Zero)
      {
        return null;
      }
      return new PqaEngine(nativeEngine);
    }
  }
}
