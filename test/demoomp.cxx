#include "omp.h"
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <CL/sycl.hpp>
using namespace sycl;

#include <chrono>


void flux(
 const double * __restrict__ Q, // Q[5+0],
 int                                          normal,
 double * __restrict__ F // F[5],
) 
{
  constexpr double gamma = 1.4;
  const double irho = 1./Q[0];
  #if Dimensions==3
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]+Q[3]*Q[3]));
  #else
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]));
  #endif

  const double coeff = irho*Q[normal+1];
  F[0] = coeff*Q[0];
  F[1] = coeff*Q[1];
  F[2] = coeff*Q[2];
  F[3] = coeff*Q[3];
  F[4] = coeff*Q[4];
  F[normal+1] += p;
  F[4]        += coeff*p;
}


double maxEigenvalue(
  const double * __restrict__ Q,
  int                                          normal
) {
  constexpr double gamma = 1.4;
  const double irho = 1./Q[0];
  #if Dimensions==3
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]+Q[3]*Q[3]));
  #else
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]));
  #endif

  const double u_n = Q[normal + 1] * irho;
  const double c   = std::sqrt(gamma * p * irho);

  double result = std::max( std::abs(u_n - c), std::abs(u_n + c) );
  return result;
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void defaultcompute(const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
  const size_t NPT=$XXX;
  for (int pidx=0;pidx<NPT;pidx++)
  {
    for (int y=0; y < numVPAIP; y++)
    for (int x=0; x < numVPAIP; x++)
    {
      for (int i=0; i<unknowns+aux; i++)
      {
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
      }
    }
  }
}


template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void computeparallel(const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
    const size_t NPT=$XXX;
#pragma omp parallel for collapse(4)
    for (int pidx=0;pidx<NPT;pidx++)
    {
      for (int y=0; y < numVPAIP; y++)
      for (int x=0; x < numVPAIP; x++)
      {
        for (int i=0; i<unknowns+aux; i++)
        {
          double *reconstructedPatch = Qin + sourcePatchSize*pidx;
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
        }
      }
    }
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void computeparallelgpu(const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
    const size_t NPT=$XXX;

#pragma omp target map(from:Qout[0:NPT*destPatchSize]) 
  {
      #pragma omp parallel for collapse(4)
    for (int pidx=0;pidx<NPT;pidx++)
    {
      for (int y=0; y < numVPAIP; y++)
      for (int x=0; x < numVPAIP; x++)
      for (int i=0; i<unknowns+aux; i++)
      {
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
      }
    }
  }
}


template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void computeparallelgpudist(const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
    const size_t NPT=$XXX;

#pragma omp target map(from:Qout[0:NPT*destPatchSize]) 
  {
#pragma omp teams distribute
    for (int pidx=0;pidx<NPT;pidx++)
    {
      #pragma omp parallel for collapse(3)
      for (int y=0; y < numVPAIP; y++)
      for (int x=0; x < numVPAIP; x++)
      for (int i=0; i<unknowns+aux; i++)
      {
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
      }
    }
  }
}



int main(int argc, char* argv[])
{


    // https://stackoverflow.com/questions/2704521/generate-random-double-numbers-in-c
    std::uniform_real_distribution<double> unif(0, 10);
    std::default_random_engine re(time(NULL));

    const size_t NPT=$XXX;
    const int numVPAIP = $YYY;
    const int unknowns=5;
    const int srcPS = (numVPAIP+2)*(numVPAIP+2)*unknowns;
    const int destPS = numVPAIP*numVPAIP*unknowns;
    
    double * Xin = new double[srcPS*NPT];
    for (int i=0;i<srcPS*NPT;i++) Xin[i] = unif(re);
#pragma omp target enter data map(to:Xin[0:srcPS*NPT])
    double * Xout = new double [destPS*NPT];

    defaultcompute<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    std::cout << "      sum should be: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";

    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    computeparallel<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    std::cout << "cpu parallel sum is: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";

    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    computeparallelgpu<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    std::cout << "gpu parallel sum is: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";

    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    computeparallelgpudist<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    std::cout << "    gpu dist sum is: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";

    auto start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      defaultcompute<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Default:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      computeparallel<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    end = std::chrono::steady_clock::now();
    std::cout << "CPU OMP:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      computeparallelgpu<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    end = std::chrono::steady_clock::now();
    std::cout << "GPU OMP:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      computeparallelgpudist<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    end = std::chrono::steady_clock::now();
    std::cout << "OMPDIST:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
   
    delete[] Xin;
    delete[] Xout;
    return 0;
}
