/*
 * BsplineAnalyticSmoother.cpp
 *
 *  Created on: Dec 25, 2012
 *  Last Modified on: Feb 11, 2013
 *  Feature: parallel BLAS and LAPACK
 *      Author: Hasib
 */

#include "BsplineAnalyticSmoother.h"

using namespace std;

BsplineAnalyticSmoother::BsplineAnalyticSmoother(double *dataset, unsigned int n_data, double *knots, unsigned int n_knot,  int degree, double smoothness, double zscore)
{
	this->degree = degree;
	this->smoothness = smoothness;
	this->zscore = zscore;
	this->n_data = n_data;
	this->n_knot = n_knot;
	this->n_coef = n_knot-degree-1;
	this->knots = knots;
	this->dataset = dataset;
	this->S = 0;
	this->smoothed_data = 0;
	this->cleaned_data = 0;

}

BsplineAnalyticSmoother::~BsplineAnalyticSmoother()
{
	if (S)
	{
		free(S);
	}
 	if (smoothed_data)
	{
		free(smoothed_data);
	}
	if (cleaned_data)
	{
		free(cleaned_data);
	}  
}

//Return basis functions for knot i and degree p given the list of knots
double BsplineAnalyticSmoother::bsplinebasis(unsigned int i, int p, double t)
{
	double this_range = 0;
    	double next_range = 0;
    	double term1 = 0;
    	double term2 = 0;

	
	if( (i+p+1) >= n_knot )
        	fprintf(stderr, "Error in i value: %d\n", i);
    
    	if (p == 0) // N_i0
    	{
		if ( (knots[i] <= t) && ( (t < knots[i+1]) || ( (t == knots[i+1]) && (t == knots[n_knot-1]) ) ) )
			return 1.0;
    		else
    			return 0.0;		
    	}
    	else    // N_ip
    	{
		this_range = knots[i+p] - knots[i];
		next_range = knots[i+p+1] - knots[i+1];
	
		if (this_range > 0)
			term1 = ((t - knots[i])/this_range) * bsplinebasis(i, p-1, t);

		if (next_range > 0)
			term2 = ((knots[i+p+1] - t)/next_range) * bsplinebasis(i+1, p-1, t);
	
		return (term1 + term2);
	}
}

// Calculate the n*k matrix where element i, j is the value at time i of j'th b-spline basis function
double* BsplineAnalyticSmoother::get_phi()
{
    int indx=0;
    double t_range;
    double t;
    double data;

    double *phi = (double*) memalign(N_ALIGN, n_data*n_coef*sizeof(double));
    
    t_range = knots[n_knot-1] - knots[0];

    //#pragma omp parallel for num_threads(n_threads) private(t, data, indx) collapse(2)
    #pragma omp parallel for private(t, data, indx) collapse(2)
    for(unsigned int i=0; i < n_data; i++)
    {
    	for(unsigned int j=0; j < n_coef; j++)
    	{
    		t = (double) (i*t_range)/(n_data-1);
    		data = bsplinebasis(j, degree, t);
    		indx = i*n_data+j;
    		phi[indx] = data;
    	}
    }
    return phi;
}

// Calculate the hat matrix S (eq.15 of Chen)
void BsplineAnalyticSmoother::calc_hatMatrix()
{
	int info;
	int ipiv[n_coef];
	//int *ipiv;
	double *phi;
	double *R;
	double *core;
	
	core = (double *) memalign(N_ALIGN, n_data*n_coef*sizeof(double));
	S = (double *) memalign(N_ALIGN, n_data*n_coef*sizeof(double));
	
	// Calculate phi and roughness
	phi = get_phi();
	R = get_roughness();

	// R = phi_t*phi + lamda*r
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_data,  n_coef,  n_coef, 1, phi, n_coef, phi, n_coef, smoothness, R, n_coef);
	
	// core = phi*inv(R)
	memcpy(core, phi, n_data*n_coef*sizeof(double));
	info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n_coef, n_coef, R, n_coef, ipiv, core, n_coef);
	
	// S = core*phi_t
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_data, n_coef, n_coef, 1.0, phi, n_coef, core, n_coef, 0, S, n_coef);

	free(core);
	free(R);
	free(phi);
}

// Calculate the smoothed data
void BsplineAnalyticSmoother::calc_smoothedData()
{
	smoothed_data = (double *) memalign(N_ALIGN, n_data*sizeof(double));
	
	calc_hatMatrix();

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_data, 1, n_coef, 1.0, S, n_coef, dataset, 1, 0, smoothed_data, 1);
}

// Compute the cleaneddata
double* BsplineAnalyticSmoother::calc_cleanedData()
{
	long indx;
	double diff;
	double df = 0;
	double total_se = 0;
	double divisor;
	double mse;
	double pe, lw, up;

	// Calculate MSE	
	calc_smoothedData();

	#pragma omp parallel for  private(diff, indx) reduction(+:total_se, df)
	for(unsigned int i=0; i < n_data; i++)
	{
		diff = dataset[i] - smoothed_data[i];
		total_se += diff*diff;
		indx = i*n_data+i;
		df += S[indx];
	}

	divisor = n_data - df;
	mse = total_se/divisor;

	// Calculate var matrix
	double *V = (double *) memalign(N_ALIGN, n_data*n_coef*sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_data, n_coef, n_coef, mse, S, n_coef, S, n_coef, 0, V, n_coef);
	
	cleaned_data = (double *) memalign(N_ALIGN, n_data*sizeof(double));
	memcpy(cleaned_data, dataset, n_data*sizeof(double));
	
	// Estimate cleaned data
	#pragma omp parallel for private(pe, lw, up)
	for(unsigned int i=0; i < n_data; i++)
	{
		pe = sqrtf( V[i*n_data+i] + mse );
		lw = smoothed_data[i] - pe*zscore;
		up = smoothed_data[i] + pe*zscore;
		if( cleaned_data[i] < lw ) 
			cleaned_data[i] = lw;
		else 
			if ( cleaned_data[i] > up )
				cleaned_data[i] = up;				
	}

	free(V);
	return cleaned_data;
}

// Calculate the n-th derivative function for the i-th basis function
// of degree.
double BsplineAnalyticSmoother::bsplinebasis_deriv(int i, int p, int n, double t)
{
	double this_range;
	double next_range;
	double term1 = 0;
	double term2 = 0;

	this_range = knots[i+p] - knots[i];
	next_range = knots[i+p+1] - knots[i+1];
	
	if (this_range > 0)
	{
		if(n == 1)
		{
			term1 = (p/this_range)*bsplinebasis(i, p-1, t);
		}
		else
		{
			term1 = (p/this_range)*bsplinebasis_deriv(i, p-1, n-1, t);
		}
	}

	if (next_range > 0)
	{
		if(n == 1)
		{
			term2 = (p/next_range)*bsplinebasis(i+1, p-1, t);
		}
		else
		{
			term2 = (p/next_range)*bsplinebasis_deriv(i+1, p-1, n-1, t);
		}
	}
		
	return (term1 - term2);
}

// Calculate the "R" matrix for degree 3
double* BsplineAnalyticSmoother::get_roughness()
{
	int a, b;
	unsigned int jmin, jmax;
	double l1, l2, l3;
	double c_0, c_1, c_2, c_3;
	double div;
	double firstd_o[n_coef];
	double firstd_n[n_coef];
	double secndd_o[n_coef];
	double secndd_n[n_coef];
	
	long indx;
	double data;

	double *R = (double*) memalign(N_ALIGN, n_data*n_coef*sizeof(double));
	memset(R, 0, n_data*n_coef*sizeof(double));

	if(degree != 3) {
		printf("\n Degree != 3: @ get_roughness");
		exit(1);
	}

	//double *temp_spline = (double*) memalign(N_ALIGN, n_knot*n_coef*sizeof(double));
	double *temp_spline = (double*) malloc(n_knot*n_coef*sizeof(double));
	
	// Parallel = OK
	#pragma omp parallel for private(indx, data) collapse(2)
	for(unsigned int i=0; i < n_knot; i++)
		for(unsigned int j=0; j < n_coef; j++)
		{
			data = bsplinebasis(j, degree, knots[i]);
			indx = i*n_coef+j;
			temp_spline[indx] = data;
			//temp_spline[i*n_knot+j] = bsplinebasis(j, degree, knots[i]);
		}

	for(unsigned int i=0; i < n_coef; i++)
	{
		firstd_o[i] = bsplinebasis_deriv(i, degree, 1, knots[0]);
		firstd_n[i] = bsplinebasis_deriv(i, degree, 1, knots[n_knot-1]);
		secndd_o[i] = bsplinebasis_deriv(i, degree, 2, knots[0]);
		secndd_n[i] = bsplinebasis_deriv(i, degree, 2, knots[n_knot-1]);
	}
			
	// memory intensive part	
	for(unsigned int i=0; i < n_coef; i++)
	{
		if(knots[i+1] > knots[i])
			c_0 = 6.0/((knots[i+3] - knots[i]) * (knots[i+2] - knots[i]) * (knots[i+1] - knots[i]));
		else
			c_0 = 0;

		div = (knots[i+3] - knots[i]) * (knots[i+2] - knots[i]);
		l1 = (div > 0) ? (1.0/div) : 0;

		div = (knots[i+3] - knots[i]) * (knots[i+3] - knots[i+1]);
		l2 = (div > 0) ? (1.0/div) : 0;

		div = (knots[i+4] - knots[i+1]) * (knots[i+3] - knots[i+1]);
		l3 = (div > 0) ? (1.0/div) : 0;

		if(knots[i+2] > knots[i+1])
			c_1 = -6.0/(knots[i+2] - knots[i+1]) * (l1 + l2 + l3);
		else
			c_1 = 0;

		div = (knots[i+3] - knots[i]) * (knots[i+3] - knots[i+1]);
		l1 = (div > 0) ? (1.0/div) : 0;

		div = (knots[i+4] - knots[i+1]) * (knots[i+3] - knots[i+1]);
		l2 = (div > 0) ? (1.0/div) : 0;

		div = (knots[i+4] - knots[i+1]) * (knots[i+4] - knots[i+2]);
		l3 = (div > 0) ? (1.0/div) : 0;

		if(knots[i+3] > knots[i+2])
			c_2 = 6.0/(knots[i+3] - knots[i+2]) * (l1+l2+l3);
		else
			c_2 = 0;

		if(knots[i+4] > knots[i+3])
			c_3 = -6.0/( (knots[i+4] - knots[i+1]) * (knots[i+4] - knots[i+2]) * (knots[i+4] - knots[i+3]) );
		else
			c_3 = 0;

		a = i-degree;
		b = i+degree+1;

		jmin = (0 > a) ? 0 : a;
		jmax = (n_coef < b) ? n_coef : b;
		
		//memory intensive part (cache optimization)
		for(unsigned int j= jmin; j < jmax; j++)
		{		
			indx = i*n_coef+j;
			data = ( secndd_n[i]*firstd_n[j] - secndd_o[i]*firstd_o[j] )\
					- c_0 * (temp_spline[(i+1)*n_coef+j] - temp_spline[i*n_coef+j])\
					- c_1 * (temp_spline[(i+2)*n_coef+j] - temp_spline[(i+1)*n_coef+j])\
					- c_2 * (temp_spline[(i+3)*n_coef+j] - temp_spline[(i+2)*n_coef+j])\
					- c_3 * (temp_spline[(i+4)*n_coef+j] - temp_spline[(i+3)*n_coef+j]);
			R[indx] = data;
		}
	}

	free(temp_spline);
	
	return R;
}

double* BsplineAnalyticSmoother::get_smoothedData()
{
	return smoothed_data;
}

//Display the cleaned data
void BsplineAnalyticSmoother::print_cleanedData()
{
	for(unsigned int i=0; i < n_data; i++)
	{
		printf("%.4f  ", cleaned_data[i]);
	}
}

extern "C" {
	BsplineAnalyticSmoother* Smoother_new(double *dataset, unsigned int n_data, double *knots, unsigned int n_knot,  int degree, double smoothness, double zscore)
	{ 
		return new BsplineAnalyticSmoother(dataset, n_data, knots, n_knot, degree, smoothness, zscore); 
	}
	void Smoother_delete(BsplineAnalyticSmoother *bsm) { delete bsm; }
	double* bsm_cleanData(BsplineAnalyticSmoother* bsm){ return bsm->calc_cleanedData(); }
	double* bsm_smoothedData(BsplineAnalyticSmoother* bsm){ return bsm->get_smoothedData(); }
}


