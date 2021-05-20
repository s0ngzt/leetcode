package interview;

import java.util.Arrays;

class Questions {

  // 面试题 16.01 - 交换数字
  public int[] swapNumbers(int[] numbers) {
    numbers[0] = numbers[0] ^ numbers[1];
    numbers[1] = numbers[0] ^ numbers[1];
    numbers[0] = numbers[0] ^ numbers[1];
    return numbers;
  }

  // 面试题 16.05 - 阶乘尾数
  public int trailingZeroes(int n) {
    int res = 0;
    while (n >= 5) {
      res += n / 5;
      n /= 5;
    }
    return res;
  }

  // 面试题 16.06 - 最小差
  public int smallestDifference(int[] a, int[] b) {
    Arrays.sort(a);
    Arrays.sort(b);
    int i = 0, j = 0;
    long ans = Long.MAX_VALUE;
    while (i < a.length && j < b.length) {
      if (a[i] == b[j]) {
        return 0;
      } else if (a[i] > b[j]) {
        ans = Math.min((long) a[i] - (long) b[j], ans);
        j++;
      } else {
        ans = Math.min((long) b[j] - (long) a[i], ans);
        i++;
      }
    }
    return (int) ans;
  }

  // 面试题 16.07 - 最大数值
  public int maximum(int a, int b) {
    long diff = (long) a - (long) b;
    return (int) (((long) a + (long) b + (diff ^ (diff >> 63)) - (diff >> 63)) >> 1);
  }

}
