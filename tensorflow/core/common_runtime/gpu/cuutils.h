#pragma once
#include "cuda.h"
#include "stdio.h"
#ifndef CUUTIL_SILENCE
#define CUUTIL_ERRCHK(x) do { \
	CUresult e = (x); \
	if(e != CUDA_SUCCESS) { \
		printf("CUDA failure at %s %d: %s\n", \
			__FILE__, __LINE__, getCuDrvErrorString(e)); \
		exit(-1);  \
	} \
} while(0)
#endif /* CUUTIL_SILENCE */

#ifdef CUUTIL_SILENCE
#define CUUTIL_ERRCHK(x) (x)
#endif /* CUUTIL_SILENCE */

static const char * getCuDrvErrorString(CUresult e) {
	
	switch(e) {
		case CUDA_SUCCESS:
			return "CUDA_SUCCESS";
		case CUDA_ERROR_INVALID_VALUE:
			return "CUDA_ERROR_INVALID_VALUE";
		case CUDA_ERROR_NOT_INITIALIZED:
			return "CUDA_ERROR_NOT_INITIALIZED";
		case CUDA_ERROR_INVALID_DEVICE:
			return "CUDA_ERROR_INVALID_DEVICE";
		case CUDA_ERROR_INVALID_CONTEXT:
			return "CUDA_ERROR_INVALID_CONTEXT";
        case CUDA_ERROR_NOT_SUPPORTED:
			return "CUDA_ERROR_NOT_SUPPORTED";
		default:
			printf("ERROR CODE = %d\n", e);
			return "getCuDrvErrorString: UNKNOWN ERROR CODE";
	}
}
/*
This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
CUDA_ERROR_OUT_OF_MEMORY = 2
The API call failed because it was unable to allocate enough memory to perform the requested operation.
CUDA_ERROR_NOT_INITIALIZED = 3
This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed.
CUDA_ERROR_DEINITIALIZED = 4
This indicates that the CUDA driver is in the process of shutting down.
CUDA_ERROR_PROFILER_DISABLED = 5
This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
Deprecated
This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via cuProfilerStart or cuProfilerStop without initialization.

CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
Deprecated
This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStart() when profiling is already enabled.

CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
Deprecated
This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStop() when profiling is already disabled.

CUDA_ERROR_STUB_LIBRARY = 34
This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.
CUDA_ERROR_NO_DEVICE = 100
This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
CUDA_ERROR_INVALID_DEVICE = 101
This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device.
CUDA_ERROR_DEVICE_NOT_LICENSED = 102
This error indicates that the Grid license is not applied.
*/
