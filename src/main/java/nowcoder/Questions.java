package nowcoder;

class Questions {

  int getMinCombination(int x) {
    int ans = 0, bit = 1;
    while (x > 9) {
      ans += (x - 9) * bit;
      x -= 9;
      bit *= 10;
    }
    return ans + x * bit;
  }

}
