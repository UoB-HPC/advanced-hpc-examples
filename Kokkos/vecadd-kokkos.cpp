#include <cstdio>
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

/* struct to hold Kokkos objects */
typedef struct
{
  // Device side pointers to arrays
  Kokkos::View<float*, Kokkos::Cuda>*  d_a;
  Kokkos::View<float*, Kokkos::Cuda>*  d_b;
  Kokkos::View<float*, Kokkos::Cuda>*  d_c;
  // HostSpace pointers with compatible layout
  Kokkos::View<float*>::HostMirror*   hm_a;
  Kokkos::View<float*>::HostMirror*   hm_b;
  Kokkos::View<float*>::HostMirror*   hm_c;
} t_kokkos;

void initialise(t_kokkos * kokkos, const int N);
void finalise(t_kokkos kokkos);

int main(int argc, char const *argv[])
{
  int N = 1024;               /* vector size */
  int num_iterations = 100000;
  t_kokkos kokkos;

  Kokkos::initialize();

  initialise(&kokkos, N);

  // Set values of a and b on the host
  for(int i = 0; i < N; i++)
  {
    (*kokkos.hm_a)[i] = 1.f;
    (*kokkos.hm_b)[i] = 2.f;
  }
  // Copy to the device buffers
  Kokkos::deep_copy(*kokkos.d_a, *kokkos.hm_a);
  Kokkos::deep_copy(*kokkos.d_b, *kokkos.hm_b);

  // Set up views
  Kokkos::View<float*, Kokkos::Cuda> a(*kokkos.d_a);
  Kokkos::View<float*, Kokkos::Cuda> b(*kokkos.d_b);
  Kokkos::View<float*, Kokkos::Cuda> c(*kokkos.d_c);

  for(int itr = 0; itr < num_iterations; itr++)
  {
    // Execute vecadd on the device
    Kokkos::parallel_for(N, KOKKOS_LAMBDA (const long index)
    {
      c[index] = a[index] + b[index];
    });
    Kokkos::fence();
  }

  // Read the result from the device buffer
  Kokkos::deep_copy(*kokkos.hm_c, *kokkos.d_c);

  // Verify the results
  int correct_results = 1;
  for(int i = 0; i < N; i++)
  {
    if(fabs((*kokkos.hm_c)[i] - 3.f) > 0.00001f)
    {
      printf("Incorrect answer at index %d\n", i);
      correct_results = 0;
    }
  }

  if(correct_results)
  {
    printf("Success!\n");
  }

  finalise(kokkos);
  Kokkos::finalize();
  return 0;
}

void initialise(t_kokkos * kokkos, const int N)
{
  kokkos->d_a = new Kokkos::View<float*, Kokkos::Cuda>("d_a", N);
  kokkos->d_b = new Kokkos::View<float*, Kokkos::Cuda>("d_b", N);
  kokkos->d_c = new Kokkos::View<float*, Kokkos::Cuda>("d_c", N);
  kokkos->hm_a = new Kokkos::View<float*, Kokkos::Cuda>::HostMirror();
  kokkos->hm_b = new Kokkos::View<float*, Kokkos::Cuda>::HostMirror();
  kokkos->hm_c = new Kokkos::View<float*, Kokkos::Cuda>::HostMirror();
  //Allocate views of the HostMirror type of a view
  *kokkos->hm_a = Kokkos::create_mirror_view(*kokkos->d_a);
  *kokkos->hm_b = Kokkos::create_mirror_view(*kokkos->d_b);
  *kokkos->hm_c = Kokkos::create_mirror_view(*kokkos->d_c);
}

void finalise(t_kokkos kokkos)
{
  delete kokkos.d_a;
  delete kokkos.d_b;
  delete kokkos.d_c;
  delete kokkos.hm_a;
  delete kokkos.hm_b;
  delete kokkos.hm_c;
}
