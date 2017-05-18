// PqaCore.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "PqaCore.h"

#include "IPqaEngine.h"


// This is an example of an exported variable
PQACORE_API int nPqaCore=0;

// This is an example of an exported function.
PQACORE_API int fnPqaCore(void)
{
    return 42;
}

// This is the constructor of a class that has been exported.
// see PqaCore.h for the class definition
CPqaCore::CPqaCore()
{
    return;
}
