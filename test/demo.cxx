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
void defaultcomputeparallel(const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
  const size_t NPT=$XXX;
#pragma omp parallel for collapse(4)
    for (int pidx=0;pidx<NPT;pidx++)
    {
      for (int y=0; y < numVPAIP; y++)
      for (int x=0; x < numVPAIP; x++)
      {
      //std::cout << "now x, y: "<< x << " " << y << "\n";
        for (int i=0; i<unknowns+aux; i++)
        {
          double *reconstructedPatch = Qin + sourcePatchSize*pidx;
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
#ifdef VERBOSE
          std::cout << "source: " << pidx*destPatchSize + sourceIndex*(unknowns+aux)+i << " -> " << pidx*destPatchSize + destinationIndex*(unknowns+aux)+i << "\n";
#endif
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
        }
      }
      //std::cout << "next patch \n";
    }
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
      //std::cout << "now x, y: "<< x << " " << y << "\n";
        for (int i=0; i<unknowns+aux; i++)
        {
          double *reconstructedPatch = Qin + sourcePatchSize*pidx;
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
#ifdef VERBOSE
          std::cout << "source: " << pidx*destPatchSize + sourceIndex*(unknowns+aux)+i << " -> " << pidx*destPatchSize + destinationIndex*(unknowns+aux)+i << "\n";
#endif
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
        }
      }
      //std::cout << "next patch \n";
    }
}


template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void fcompute(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const double dt =0.5;
  const size_t NPT=$XXX;


  Q.submit([&](handler &cgh)
  {

#ifdef VERBOSE
    sycl::stream out(100000, 768, cgh);
#endif
    cgh.parallel_for_work_group(range<2>{NPT, numVPAIP * numVPAIP}, {$GX, $GY}, [=](group<2> grp)
    {
      const size_t pidx=grp[0];
      double *reconstructedPatch = Qin + sourcePatchSize*pidx;
      grp.parallel_for_work_item([&](auto idx)
      {
          const size_t i=idx.get_global_id(1);
          auto [y,x] = glob2loc(i, numVPAIP);
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          for (int i=0; i<unknowns+aux; i++)
          { 
#ifdef VERBOSE
            out << "SOURCE: " << sourceIndex*(unknowns+aux)+i << " -> " << pidx*destPatchSize + destinationIndex*(unknowns+aux)+i << "\n";
#endif
            Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
          }
      });
    });
  }).wait();
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void fcompute2(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const double dt =0.5;
  const size_t NPT=$XXX;


  Q.submit([&](handler &cgh)
  {

#ifdef VERBOSE
    sycl::stream out(100000, 768, cgh);
#endif
    cgh.parallel_for_work_group(range<2>{NPT, numVPAIP * numVPAIP * (unknowns+aux)}, {$GX, $GY}, [=](group<2> grp)
    {
      const size_t pidx=grp[0];
      double *reconstructedPatch = Qin + sourcePatchSize*pidx;
      grp.parallel_for_work_item([&](auto idx)
      {
          const size_t i=idx.get_global_id(1);
          auto [z,y,x] = glob2loc(i, numVPAIP, numVPAIP);
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
#ifdef VERBOSE
            out << "sourCE: " << sourceIndex*(unknowns+aux)+z << " -> " << pidx*destPatchSize + destinationIndex*(unknowns+aux)+z << "\n";
#endif
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+z] = reconstructedPatch[sourceIndex*(unknowns+aux)+z];
      });
    });
  }).wait();
}

// We like this one as it is fast on cpu and gpu
template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void fcompute3(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const size_t NPT=$XXX;

  Q.submit([&](handler &cgh)
  {
#ifdef VERBOSE
    sycl::stream out(100000, 768, cgh);
#endif
    cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, numVPAIP}, {1, numVPAIP, numVPAIP}}, [=](nd_item<3> item)
    //cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, numVPAIP}, {1, numVPAIP, numVPAIP}}, [=](nd_item<3> item)
    {
        const size_t pidx=item.get_global_id(0);//[0];
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        const size_t x=item.get_global_id(1);
        const size_t y=item.get_global_id(2);
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        for (int i=0; i<unknowns+aux; i++)
        { 
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
void fcompute4(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const size_t NPT=$XXX;


  Q.submit([&](handler &cgh)
  {

#ifdef VERBOSE
    sycl::stream out(100000, 768, cgh);
#endif
    cgh.parallel_for_work_group(range<3>{NPT, numVPAIP, numVPAIP}, {$GX, $GY, $GZ}, [=](group<3> grp)
    {
      const size_t pidx=grp[0];
      double *reconstructedPatch = Qin + sourcePatchSize*pidx;
      grp.parallel_for_work_item([&](auto idx)
      {
          const size_t x=idx.get_global_id(1);
          const size_t y=idx.get_global_id(2);
          //auto [y,x] = glob2loc(i, numVPAIP);
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          for (int i=0; i<unknowns+aux; i++)
          { 
#ifdef VERBOSE
            out << "SOURCE: " << sourceIndex*(unknowns+aux)+i << " -> " << pidx*destPatchSize + destinationIndex*(unknowns+aux)+i << "\n";
#endif
            Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
          }
      });
    });
  }).wait();
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void hcompute(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool wtf)
{
  const size_t NPT=$XXX;

  Q.submit([&](handler &cgh)
  {
    //sycl::stream out(100000, 768, cgh);
    //cgh.parallel_for_work_group(range<3>{NPT, numVPAIP, numVPAIP}, {1,1,1}, [=](group<3> grp)
    cgh.parallel_for_work_group(range<3>{NPT, numVPAIP, numVPAIP},  [=](group<3> grp)
    {
      const size_t pidx=grp[0];
      double *reconstructedPatch = Qin + sourcePatchSize*pidx;
      grp.parallel_for_work_item([&](auto idx)
      {
          const size_t y=idx.get_global_id(1);
          const size_t x=idx.get_global_id(2);
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          for (int i=0; i<unknowns+aux; i++)
          { 
            Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
          }
      });
    });
  }).wait();
 

  //Q.submit([&](handler &cgh)
  //{
    ////sycl::stream out(100000, 768, cgh);
    //cgh.parallel_for_work_group(range<3>{NPT, numVPAIP/2, numVPAIP},  [=](group<3> grp)
    ////cgh.parallel_for_work_group(range<3>{NPT, numVPAIP/2, numVPAIP}, {1,1,1}, [=](group<3> grp)
    //{
      //const size_t pidx=grp[0];
      //double *reconstructedPatch = Qout + sourcePatchSize*pidx;
      
      //const double h0 = 0.1;
      //const double dx = 0.1;//volumeH(0);


      //for (int shift = 0; shift < 2; shift++)
      //{

        //// Normal 0
        //grp.parallel_for_work_item([&](auto idx)
        //{
            //const int normal = 0;
            //const size_t x=shift + 2*idx.get_global_id(1);
            //const size_t y=idx.get_global_id(2);

            //int leftVoxelInPreimage  = x +      (y + 1) * (2 + numVPAIP);
            //int rightVoxelInPreimage = x + 1  + (y + 1) * (2 + numVPAIP);
            //double * QL = reconstructedPatch + leftVoxelInPreimage  * (unknowns + aux);
            //double * QR = reconstructedPatch + rightVoxelInPreimage * (unknowns + aux);
            
            //double fluxFL[unknowns], fluxFR[unknowns], fluxNCP[unknowns];
                
            ////if (not skipFluxEvaluation)
            ////{
            //flux(QL, normal, fluxFL);
            //flux(QR, normal, fluxFR);
            ////}

            //double lambdaMaxL = maxEigenvalue(QL,normal);
            //double lambdaMaxR = maxEigenvalue(QR,normal);
            //double lambdaMax  = std::max( lambdaMaxL, lambdaMaxR );
            
            //int leftVoxelInImage     = x - 1 + y * numVPAIP;
            //int rightVoxelInImage    = x     + y * numVPAIP;
                
            //for (int unknown = 0; unknown < unknowns; unknown++)
            //{
              //if (x > 0)
              //{
                //double fl = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                ////if (not skipFluxEvaluation) 
                  //fl +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown];
                //Qout[pidx*destPatchSize + leftVoxelInImage * (unknowns + aux) + unknown]  -= dt / h0 * fl;
              //}
              //if (x < numVPAIP/2)
              //{
                //double fr = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                ////if (not skipFluxEvaluation)
                  //fr +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown]; 
                //Qout[pidx*destPatchSize + rightVoxelInImage * (unknowns + aux) + unknown] += dt / h0 * fr;
              //}
            //}
        //}); // Do we have an implicit barrier here or do we need to sync?
      
        //// Normal 1 (NOTE: I simply swap x and y here, needs checked)
        //grp.parallel_for_work_item([&](auto idx)
        //{
            //const int normal = 1;
            //const size_t y=shift + 2*idx.get_global_id(1);
            //const size_t x=idx.get_global_id(2);
           
            //int lowerVoxelInPreimage = x + 1  +       y * (2 + numVPAIP);
            //int upperVoxelInPreimage = x + 1  + (y + 1) * (2 + numVPAIP);
            //int lowerVoxelInImage    = x      + (y - 1) *      numVPAIP ;
            //int upperVoxelInImage    = x      +       y *      numVPAIP ;

            //double* QL = reconstructedPatch + lowerVoxelInPreimage * (unknowns + aux);
            //double* QR = reconstructedPatch + upperVoxelInPreimage * (unknowns + aux);

            
            //double fluxFL[unknowns], fluxFR[unknowns], fluxNCP[unknowns];
                
            ////if (not skipFluxEvaluation)
            ////{
            //flux(QL, normal, fluxFL);
            //flux(QR, normal, fluxFR);
            ////}

            //double lambdaMaxL = maxEigenvalue(QL,normal);
            //double lambdaMaxR = maxEigenvalue(QR,normal);
            //double lambdaMax  = std::max( lambdaMaxL, lambdaMaxR );
                
            //for (int unknown = 0; unknown < unknowns; unknown++)
            //{
              //if (y > 0)
              //{
                //double fl = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                ////if (not skipFluxEvaluation) 
                  //fl +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown];
                  //Qout[pidx*destPatchSize + lowerVoxelInImage * (unknowns + aux) + unknown] -= dt / h0 * fl;
              //}
              //if (y < numVPAIP/2)
              //{
                //double fr = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                ////if (not skipFluxEvaluation)
                  //fr +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown]; 
                  //Qout[pidx*destPatchSize + upperVoxelInImage * (unknowns + aux) + unknown] += dt / h0 * fr;
              //}
            //}
        //}); // Do we have an implicit barrier here or do we need to sync?
      
      //}
      //});
  //}).wait();
}


template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void qcompute(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
  const size_t NPT=$XXX;
  const double dt =0.5;

  Q.submit([&](handler &cgh) 
  {
      //sycl::stream out(100000, 256, cgh);
      cgh.parallel_for(
      sycl::range<3>{NPT,numVPAIP, numVPAIP}, [=] (auto it) 
      { 
      
         const size_t pidx=it[0];
         const size_t x=it[1];
         const size_t y=it[2];
         double *reconstructedPatch = Qout + sourcePatchSize*pidx;
         
         int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
         int destinationIndex = y*numVPAIP + x;
         //out << "s: " << sourceIndex << " d: " << destinationIndex << "\n";
         //for (int i=0; i<unknowns+aux; i++) out << pidx*destPatchSize + destinationIndex*(unknowns+aux)+i << "\n";
         for (int i=0; i<unknowns+aux; i++)// out << sourceIndex*(unknowns+aux)+i << "\n";
         { 
           Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
         }

      });
  });
  Q.wait();
  
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void fqcompute(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
  const size_t NPT=$XXX;
  const double dt =0.5;

  Q.submit([&](handler &cgh) 
  {
      //sycl::stream out(100000, 256, cgh);
      cgh.parallel_for(
      sycl::range<1>{NPT*numVPAIP*numVPAIP}, [=] (auto it) 
      { 
      
         const size_t g=it[0];
         auto [pidx,x,y] = glob2loc(g, numVPAIP, numVPAIP);
         double *reconstructedPatch = Qout + sourcePatchSize*pidx;
         
         int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
         int destinationIndex = y*numVPAIP + x;
         //out << "s: " << sourceIndex << " d: " << destinationIndex << "\n";
         //for (int i=0; i<unknowns+aux; i++) out << pidx*destPatchSize + destinationIndex*(unknowns+aux)+i << "\n";
         for (int i=0; i<unknowns+aux; i++)// out << sourceIndex*(unknowns+aux)+i << "\n";
         { 
           Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
         }

      });
  });
  Q.wait();
  
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
    defaultcompute<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    std::cout << "sum should be: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";

    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    defaultcomputeparallel<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    std::cout << "sum is: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";
    //for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    //fcompute<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";
    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    fcompute3<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";

    for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    hcompute<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";


    auto start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      fcompute<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    auto end = std::chrono::steady_clock::now();
    std::cout << "PAR FOR WG with glob2loc: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;

    start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      defaultcomputeparallel<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    end = std::chrono::steady_clock::now();
    std::cout << "    OMP PARALLEL FOR CPU:" <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      fcompute3<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    end = std::chrono::steady_clock::now();
    std::cout << "              NDRANGE<3>:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[1]);i++) 
      hcompute<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    end = std::chrono::steady_clock::now();
    std::cout << "     PAR FOR WG RANGE<3>: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;

    free(Xin, Q);
    free(Xout, Q);

    return 0;
}
