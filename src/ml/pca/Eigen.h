// Eigen.h: interface for the CEigen class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_EIGEN_H__CAE598AE_AEDE_4100_9B2A_0FDDE9442166__INCLUDED_)
#define AFX_EIGEN_H__CAE598AE_AEDE_4100_9B2A_0FDDE9442166__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CEigen  
{
public:
	CEigen();
	virtual ~CEigen();
	int cvJacobiEigens_64d(double* A,double* V,double* E,int n,double  eps );
	int cvJacobiEigens_32f (float* A,float* V,float* E,int n,float  eps );

};

#endif // !defined(AFX_EIGEN_H__CAE598AE_AEDE_4100_9B2A_0FDDE9442166__INCLUDED_)
