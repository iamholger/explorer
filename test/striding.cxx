#include "omp.h"
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <CL/sycl.hpp>
using namespace sycl;

#include <chrono>


#ifdef NOGPU
static queue Q(cpu_selector{});
#else
static queue Q(gpu_selector{});
#endif

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


// 2D
std::tuple<int, int> glob2loc(const int g, const int J)
{
    return std::make_tuple(g/J,g%J);
}

// 3D
std::tuple<int, int, int> glob2loc(const int g, const int J, const int K)
{
    int i = g/J/K;
    return std::make_tuple(i, (g-i*J*K)/J, g%K);
}



template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void strider(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const size_t NPT=$XXX;

  Q.submit([&](handler &cgh)
  {
    cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, unknowns+aux}, {$GX, $GY, $GZ}}, [=](nd_item<3> item)
    {
        const size_t pidx=item.get_global_id(0);//[0];
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        const size_t x=item.get_global_id(1);
        const size_t i=item.get_global_id(2);
        for (int y=0; y < numVPAIP; y++)
        {
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
        }
    });
  }).wait();
}


template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void strider2(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const size_t NPT=$XXX;

  Q.submit([&](handler &cgh)
  {
    cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, (unknowns+aux)*numVPAIP}, {$GX, $GY, $GZ}}, [=](nd_item<3> item)
    {
        const size_t pidx=item.get_global_id(0);//[0];
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        const size_t x=item.get_global_id(1);
        auto [i,y] = glob2loc(item.get_global_id(2), numVPAIP);
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
    });
  }).wait();
}

int main(int argc, char* argv[])
{
    std::cout << "  Using SYCL device: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

    Q.submit([&](handler &cgh)
    { 
       cgh.single_task([=]() {});
    });

    // https://stackoverflow.com/questions/2704521/generate-random-double-numbers-in-c
    std::uniform_real_distribution<double> unif(0, 10);
    std::default_random_engine re(time(NULL));

    const size_t NPT=$XXX;
    const int numVPAIP = $YYY;
    const int unknowns=5;
    const int srcPS = (numVPAIP+2)*(numVPAIP+2)*unknowns;
    const int destPS = numVPAIP*numVPAIP*unknowns;
    
    auto Xin  = malloc_shared<double>(srcPS*NPT, Q);
    for (int i=0;i<srcPS*NPT;i++) Xin[i] = unif(re);
    auto Xout = malloc_shared<double>(destPS*NPT, Q);

    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    strider<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    std::cout << "sum should be: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";

    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    strider2<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";

    
    auto start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      strider<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      strider2<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;

    free(Xin, Q);
    free(Xout, Q);

    return 0;
}
