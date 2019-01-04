@REM Change to Debug when you need to debug the engine
SET Configuration=Release
FOR %%e in (dll pdb) DO (
  FOR %%n in (PqaCore SRPlatform) DO (
    @REM This assumes the relative path to ProbQA engine
    XCOPY /Y /F ..\..\..\..\ProbQA\x64\%Configuration%\%%n.%%e .
  )
)