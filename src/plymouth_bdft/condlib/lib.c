#include <math.h> // which one first?
#include <complex.h>

#include <stddef.h>
#include <stdlib.h>
#include <fftw3.h>
#include <malloc.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#define TAU 6.28318530717958623199592693708837032318115234375
#define ROOT_TAU 2.506628274631000241612355239340104162693023681640625

#define logout(format, ...) fprintf(stderr, "\033[3;33mcondlib(%s:%03d)::\033[0m " format, __FUNCTION__, __LINE__, ##__VA_ARGS__)

static int mcount = 0;
void* alloc(size_t size){
  logout("Allocation #%d, size: %zu\n", ++mcount, size);
  void* pntr = malloc(size);
  if (pntr == NULL){
    logout("Failed to allocate!\n");
    exit(1);
  }
  return pntr;
}

void panic(char* msg){
  printf("PANIC!: %s\n", msg);
  exit(1);
}


// //  =============================================================================================
// //                                       Plotting Code
// //  =============================================================================================

// struct plot_2d_config {
//   size_t N1; size_t N2;
//   char* title;
//   double* x;
//   double* y;
//   enum {REAL_IMAGE, COMP_IMAGE} type;
//   union {double complex* comp_image; double* real_image;};
//   union {double (*map_c)(double complex); double (*map_r)(double);};
// };

// double identity(double x) {return x;}

// void plot(int _N, double x[], complex double fx[]) {
//     double max_r = -1e300;
//     double max_i = -1e300;
//     double min_r = 1e300;
//     double min_i = 1e300;
//     for (int i = 0; i<_N; i++){
//      max_r = creal(fx[i]) > max_r ? creal(fx[i]) : max_r;
//      max_i = cimag(fx[i]) > max_i ? cimag(fx[i]) : max_i;

//      min_r = creal(fx[i]) < min_r ? creal(fx[i]) : min_r;
//      min_i = cimag(fx[i]) < min_i ? cimag(fx[i]) : min_i;
//    }

//    printf("range [%f, %f] + i [%f, %f]\n", min_r, max_r, min_i, max_i);

//     for (int i = 0; i<_N; i++){
//     printf("\033[35m%6.3f|\033[0m", x[i]);

//     bool printed_r = false;
//     bool printed_i = false;
//     int width = 35;
//     for (int j = 0; j <= 2*width + 1; j++){
//       if ((j > width*((creal(fx[i]) - min_r)/(max_r - min_r))*2) & !printed_r) {
//         printed_r = true;
//         printf("\033[32mr\033[0m");
//       }
//       else if (j == width) printf("\033[35m¦\033[0m");
//       else if ((j > width*((cimag(fx[i]) - min_i)/(max_i - min_i))*2) & !printed_i) {
//         printed_i = true;
//         printf("\033[31mc\033[0m");
//       }
//       else printf(" ");
//     }
//     printf(" ¦  %3.1f + i %f3.1\n",creal(fx[i]), cimag(fx[i]));
//   }
// }



// void plot_2d(struct plot_2d_config config){
//   int N1 = config.N1;
//   int N2 = config.N2;
//   double* x = config.x;
//   double* y = config.y;


//   if (N1 == 0) panic("Called with invalid arguments, one dimention must be provieded");
//   if (N2 == 0) N2 = N1;


//   if (x == NULL){
//     x = alloc(N1*sizeof(double));
//     for (int i=0; i<N1; i++) x[i] = i;
//   }

//   if (y == NULL){
//     y = alloc(N2*sizeof(double));
//     for (int i=0; i<N2; i++) y[i] = i;
//   }

//  if (config.type == COMP_IMAGE) if (config.map_c == NULL) config.map_c = creal;
//   if (config.type == REAL_IMAGE) if (config.map_r == NULL) config.map_r = identity;

//   char* colors[10] = {
//     "38;5;18", // 0
//     "38;5;20", // 1
//     "38;5;39", // 2
//     "38;5;37", // 3
//     "38;5;35", // 4
//     "38;5;70", // 5
//     "38;5;142", // 6
//     "38;5;214", // 7
//     "38;5;208", // 8
//     "38;5;196", // 9
//   };


//   double min = 1e100;
//   double max = -1e100;

//   for (int i=0; i<N1; i++)
//     for(int j=0; j<N2; j++){
//       const double point = (config.type == REAL_IMAGE)? config.map_r(config.real_image[j+N1*(i)])
//                                                       : config.map_c(config.comp_image[j+N1*(i)]);
//       max = point > max? point: max;
//       min = point < min? point: min;
//     }
     
//   printf("scaile:\n");
//   for (int n=0; n<10; n++){
//     printf("\t\033[%sm%d\033[0m -- %6.3E\n", colors[n], n, (double)n/9 *(max-min) + min );
//   }
//   printf("\nbin width = %E\n", (max-min)/9);
//   printf("\n");

//   printf("                       "); int title_len =  printf("%s\n", config.title);
//   printf("                       "); for (int i=0; i<title_len; i++) printf("-");
//   printf("\n");

//   for (int i=0; i<N1; i++){
//     for(int j=0; j<N2; j++){
//       const unsigned point = round(9*(
//                                  ((config.type == REAL_IMAGE)? config.map_r(config.real_image[j+N1*(i)])
//                                                       : config.map_c(config.comp_image[j+N1*(i)]))
//                                  -min)/(max-min));
//       printf(" \033[%sm%d\033[0m", colors[point], point);
//       }
//     printf("\n");
//    }
// }

//  =============================================================================================
//                                       Library Code
//  =============================================================================================
   
const char* version(){
  return "Cond Lib built at "__TIME__ " on "  __DATE__ " from the source file. File Hash " __FILE_HASH__ "." ;
};
   
  //   output array is 4 dimantionsal
  //  --------------------------------
  // 
  //                ,----------| a index
  //               /   ,-------| b index
  //   ZpE  = 2 × N × N × 2;
  //           \           `---| Real & imagenary index
  //            `--------------| E & Eperp index
  // 
  // 
  //  I dont use  complex.h because python's ctypes
  //  does not have good interop for it

void calculate_nabla(double* nabla_out,
                     const double* image_in,
                     const double ai[2][2],
                     const size_t N,
                     const double L){
 //   calculate nabla along indexes via finite difference.
 //      - central difference in the bulk
 //      - forward difference on the left/top
 //      - backwards difference on the right/bottom
 //  input shape  N × N
 //  output shape 2 × N × N
     
  //            ,-> a
  // epsilon = N × N
  //                `-> b

  //                  ,-> alpha
  // nabla epsilon  = 2 × N × N
  //                       \  `-> b
  //                        `---> a (a>=0)

  logout("Calculateing NU\n");
  for (size_t a = 0; a < N; a++)
    for (size_t b = 0; b < N; b++){
      // logout("\t (a,b) = (%d,%d)\n", a,b);
      double diff1, diff2;
      if (a == 0){ // top row
        diff1 = (image_in[b + N*(a+1)] - image_in[b + N*(a)]) * N/L;
      } else if (a == N-1){ // bottm row
        diff1 = (image_in[b + N*a]     - image_in[b + N*(a-1)]) * N/L;
      } else { // middle of collumn
        diff1 = (image_in[b + N*(a+1)] - image_in[b + N*(a-1)])/2.0 * N/L;
      }

      if (b == 0) { // left column
        diff2 = (image_in[b+1 + N*a] - image_in[b + N*a]) * N/L;
      } else if (b == N-1){ // right column
        diff2 = (image_in[b + N*a] - image_in[b-1 + N*a]) * N/L;
      } else { // middle of row
        diff2 = (image_in[(b + 1) + N*a] - image_in[(b-1) + N*a])/2.0 * N/L;
      }

      nabla_out[b + N*(a + N*(0))] = diff1 * ai[0][0] + diff2 * ai[1][0] ;
      nabla_out[b + N*(a + N*(1))] = diff1 * ai[0][1] + diff2 * ai[1][1] ;
  }
logout("\n");
}


inline double hexegon_area(const double ai[2][2]){
  // A = sqrt( (|a1||a2|)² - (a1•a2)² )
  return sqrt( (ai[0][0] * ai[0][0] + ai[0][1] * ai[0][1]) * (ai[1][0] * ai[1][0] + ai[1][1] * ai[1][1]) // |a1|² * |a2|²
             - (ai[0][0] * ai[1][0] + ai[0][1] * ai[1][1]) * (ai[0][0] * ai[1][0] + ai[0][1] * ai[1][1]) // - (a1 • a2)²
    ); 
}

inline void calculate_nuk(double complex* nuk, const double* epsilon, const double beta, const double mu, const size_t N){
  for (size_t l=0; l<N; l++)
    for (size_t m=0; m<N; m++){
      nuk[m + N*(l)] = 1/(1 + exp(-beta *(epsilon[m + N*(l)] + mu)));
  }
}

void calculate_nu(
  double complex* nu_ab,
  const double beta,
  const double mu,
  const double* epsilon,
  const size_t N ){

    double complex* nuk = fftw_alloc_complex(N*N*sizeof(double complex));
    calculate_nuk(nuk, epsilon, beta, mu, N);

    // Im not sure if this is the right sign
    fftw_plan plan = fftw_plan_dft_2d(N,N, nuk, nu_ab, -1, FFTW_ESTIMATE);
    fftw_execute(plan);

    fftw_destroy_plan(plan);
    fftw_free(nuk);

    for (size_t i=0; i< N*N; i++) nu_ab[i] /= N*N;
    // maybe a 2pi?
}

void calculate_nu_slow(
  double complex* nu_ab,
  const double beta,
  const double mu,
  const double* epsilon,
  const size_t N ){

  for (int a=0; a<N; a++)
    for (int b=0; b<N; b++){
      nu_ab[b + N*a] = 0;
      for (int i=0; i<N; i++)
        for (int j=0; j<N; j++) {
          nu_ab[b + N*a] += cexp(-I*TAU*(a*i + b*j)/N) * 1./(1.+exp(-beta*(epsilon[j + N*i] + mu))) /N/N;
      }
  }
}


// I noticed that the results of this seem to be a constant
// Ive imported the plotting function now so lets see it
void nuk_reconstruction(
  double complex* nuk,
  const complex double* nu_ab,
  const double a[2][2],
  const size_t N
){
  // TODO: Its the slow index that is trunkated, not the fast index....
  // Checks the deffinition of nu_ab,
  // that is that when sumed against phases that match its indexes
  // nu_k is reproduec. No use of exsternal fft libraries used.
  //
  // Assume the outptu buffer (nuk) is zeroed already
  // The output is complex but the real part should be zero.

  for (unsigned i=0; i<N; i++)
    for (unsigned j=0; j<N; j++) {
      nuk[j+N*i] = 0;
      for (unsigned a=0; a<N; a++) 
        for (unsigned b=0; b<N; b++) {
              nuk[j+N*i] += nu_ab[b + N*a] * cexp(I*TAU*(a*i + b*j)/N);
      }
  }

  double x[N];
  complex double fx[N];
  for (int i=0; i<N; i++){
    x[i] = i;
    fx[i] = nuk[N*i];
  }


  // plot_2d(
  //   (struct plot_2d_config){
  //   .N1 = N,
  //   .type = COMP_IMAGE,
  //   .map_c = creal,
  //   .title = "reconstructed nuk (real part)",
  //   .comp_image = nuk,
  //   }
  // );

  // plot(N,x,fx);

  logout("N = %zu\n", N);

}

//                     ,---| beta
//                    /
//                   /  ,-| tearms reflected in b
// Gamma_ab^beta ~= 2 × 2 × N × N 
//                          \---/
//                           a b

void calculate_gamma(double complex* gamma,
                     const double* epsilon,
                     const double ai[2][2],
                     const size_t N, const double L){
    logout("C gamma calc called\n");

    //          ,-> sign of the small index
    // gamma = 2 × N × N
    //              \   `-> small index
    //               `-> large index

    //            ,-> a
    // epsilon = N × N
    //                `-> b

    {
      logout("\033[32mabout to calculate nabla\n");
      //                  ,-> alpha
      // nabla epsilon  = 2 × N × N
      //                       \  `-> b
      //                        `---> a (a>=0)
      double* nabla_epsilon = alloc(2*N*N*sizeof(double));
      calculate_nabla(nabla_epsilon, epsilon, ai, N, L);

      logout("about to do the fft\n");
      fftw_plan plan = fftw_plan_dft_r2c_2d(N, N, nabla_epsilon, gamma, FFTW_ESTIMATE);
      fftw_execute(plan);

      plan = fftw_plan_dft_r2c_2d(N, N, nabla_epsilon + N*N, gamma, FFTW_ESTIMATE);
      fftw_destroy_plan(plan);
      logout("fft complete\n");

      free(nabla_epsilon);
    }

    logout("Normalizeing\n");
    const double A = hexegon_area(ai);
    for (size_t a = 0; a < N; a++)
      for (size_t b = 0; b < N; b++) {
        gamma[b + N*(a+N*0)] /= A*N*N;
        gamma[b + N*(a+N*1)] /= A*N*N;
    }
      
}

void calculate_sigma(double *sigma_tau,
                     const double* epsilon,
                     const double E,
                     const double ai[2][2],
                     const double Ehat[2],
                     const size_t N,
                     const double etau_hbar){

   logout("Sigma calculation started\n");
   // nu is the FFT of a real valued 2D array of shape N×N,
   // thus the output is a complex valued 2D of shape (N/2 +1)×N
   logout("calculating nu\n");
   double complex* nu = alloc(N*N*sizeof(double complex));
   calculate_nu(nu,1.0, 0.0, epsilon, N);

   logout("calculating Gamma\n");
   double complex* gamma = alloc(2*N*N*sizeof(double complex));
   calculate_gamma(gamma, epsilon, ai, N, 1.0/N); // need a better delta

   const double Ehat_p[2] = {Ehat[1], -Ehat[0]};
   const double Ehat_b0_dot_a1 = Ehat[0] * ai[0][0] + Ehat[1] * ai[0][1];
   const double Ehat_b0_dot_a2 = Ehat[0] * ai[1][0] + Ehat[1] * ai[1][1];
   const double Ehat_b1_dot_a1 = Ehat_p[0] * ai[0][0] + Ehat_p[1] * ai[0][1];
   const double Ehat_b1_dot_a2 = Ehat_p[0] * ai[1][0] + Ehat_p[1] * ai[1][1];

   //                      ,-> beta
   const double Ehat_dot_a[2][2] = {
   //                          `-> i
     { // Ehat dot a0,  Ehat dot a1
       Ehat[0] * ai[0][0] + Ehat[1] * ai[0][1],
       Ehat[0] * ai[1][0] + Ehat[1] * ai[1][1]
     },
     { // Ehat_p dot a0, Ehat_p dot a1
       Ehat_p[0] * ai[0][0] + Ehat_p[1] * ai[0][1],
       Ehat_p[0] * ai[1][0] + Ehat_p[1] * ai[1][1]
     }
   };
   
 logout("Starting main sigma tau loop\n");

  for (unsigned alpha = 0; alpha < 2; alpha++)
    for (unsigned beta = 0; beta < 2; beta++)
      for (int a = 0; a < N; a++){
        logout("outer_loop = %d   \r", a);
        for (int b = 0; b < N/2+1; b++) { // <- this should be over the full range
          sigma_tau[b + N*(a + N*(beta + 2*alpha))] = 0;
          for (int sb = 0; sb < 2; sb++) {
          //  reflection is not done correctly
          //                        a b
          //                       /---\
          // sigma_tau =   2 × 2 × N × N 
          //               \---/       
          //             alpha beta       
          // 
          // I need to understand the negetive sign
          sigma_tau[b + N*(a + N*(beta + 2*alpha))] += 2*creal(
                                                   gamma[b + N*(a + N*(alpha))] // gamma has the same indexing structure, with an additional large demention for the alpha index
                                                 * nu[b + N*a] // this is nu_ab, a has negetive indexies but b does not, thus the rante is a->N, b->N/2+1
                                                 * I * etau_hbar * (
                                                     a * Ehat_dot_a[beta][0] + (1-2*sb)*b *Ehat_dot_a[beta][1]
                                                 ) * cpow(
                                                          1 + I * etau_hbar *(
                                                                a * Ehat_dot_a[0][0] + (1-2*sb)*b * Ehat_dot_a[0][1]
                                                            ),
                                                          -2)
                                                 );

      }
    }
  }

  free(nu);
  free(gamma);
  logout("\n");
}
