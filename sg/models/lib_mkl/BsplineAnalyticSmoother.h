/*
 * BsplineAnalyticSmoother.cpp
 *
 * Created on: Dec 25, 2012
 * Last Modified on: Feb 7, 2013
 * Feature: parallel BLAS and LAPACK
 * Author: Hasib
 *
 * */

#ifndef BSPLINEANALYTICSMOOTHER_H_
#define BSPLINEANALYTICSMOOTHER_H_

#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif
	#include <math.h>
	#include <stdlib.h>	
	#include <string.h>
	#include <malloc.h>
	#include <omp.h>
	#include <mkl_lapacke.h>
	#include <mkl.h>

#ifdef __cplusplus
}
#endif

#define N_ALIGN (size_t)64

class BsplineAnalyticSmoother {

	private:
		int degree;
		unsigned int n_data;
		unsigned int n_knot;
		unsigned int n_coef;
		double smoothness;
		double zscore;
		double *knots;
		double *dataset;
		double *S;
		double *smoothed_data;
		double *cleaned_data;

		double bsplinebasis(unsigned int i, int p, double t);
		double bsplinebasis_deriv(int i, int p, int n, double t);
		double* get_phi();
		double* get_roughness();
		void calc_hatMatrix();

	public:
		BsplineAnalyticSmoother(double *dataset, unsigned int n_data, double *knots, unsigned int n_knot,  int degree, double smoothness, double zscore);
		virtual ~BsplineAnalyticSmoother();
		void calc_smoothedData();
		double* calc_cleanedData();
		void print_cleanedData();
		double *get_smoothedData();

};

#endif /* BSPLINEANALYTICSMOOTHER_H_ */
