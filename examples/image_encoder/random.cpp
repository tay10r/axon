#include "random.h"

#include <random>

std::mt19937 rng(0);

int
randint(int lo, int hi)
{
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(rng);
}
