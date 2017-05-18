// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the PQACORE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// PQACORE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef PQACORE_EXPORTS
#define PQACORE_API __declspec(dllexport)
#else
#define PQACORE_API __declspec(dllimport)
#endif

// This class is exported from the PqaCore.dll
class PQACORE_API CPqaCore {
public:
	CPqaCore(void);
	// TODO: add your methods here.
};

extern PQACORE_API int nPqaCore;

PQACORE_API int fnPqaCore(void);
