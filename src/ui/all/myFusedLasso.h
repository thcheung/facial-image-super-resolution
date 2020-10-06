/*
 * MATLAB Compiler: 6.5 (R2017b)
 * Date: Tue Jan  1 17:36:46 2019
 * Arguments:
 * "-B""macro_default""-B""csharedlib:myFusedLasso""-W""lib:myFusedLasso""-T""li
 * nk:lib""myFusedLeastR.m""-d"".\output"
 */

#ifndef __myFusedLasso_h
#define __myFusedLasso_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" {
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_myFusedLasso_C_API 
#define LIB_myFusedLasso_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_myFusedLasso_C_API 
bool MW_CALL_CONV myFusedLassoInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_myFusedLasso_C_API 
bool MW_CALL_CONV myFusedLassoInitialize(void);

extern LIB_myFusedLasso_C_API 
void MW_CALL_CONV myFusedLassoTerminate(void);

extern LIB_myFusedLasso_C_API 
void MW_CALL_CONV myFusedLassoPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_myFusedLasso_C_API 
bool MW_CALL_CONV mlxMyFusedLeastR(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_myFusedLasso_C_API bool MW_CALL_CONV mlfMyFusedLeastR(int nargout, mxArray** w, mxArray* A, mxArray* Y, mxArray* lambda, mxArray* lambda2);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif
