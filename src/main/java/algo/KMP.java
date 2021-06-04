package algo;

public class KMP {

  /**
   * 暴力破解法
   *
   * @param ts 主串
   * @param ps 模式串
   * @return 如果找到，返回在主串中第一个字符出现的下标，否则返回 -1
   */

  public static int bf(String ts, String ps) {

    char[] t = ts.toCharArray();
    char[] p = ps.toCharArray();

    int i = 0; // 主串的位置
    int j = 0; // 模式串的位置

    while (i < t.length && j < p.length) {
      if (t[i] == p[j]) { // 当两个字符相同，就比较下一个
        i++;
        j++;
      } else {
        i = i - j + 1; // 一旦不匹配，i 后退
        j = 0; // j 归 0
      }
    }

    if (j == p.length) {
      return i - j;
    } else {
      return -1;
    }
  }

  /**
   * 计算每一个位置j对应的k，所以用一个数组 next 保存
   * <p>
   * next[j] = k，表示当T[i] != P[j]时，j指针的下一个位置
   *
   * @param ps 模式串
   * @return next[] 数组
   */
  public static int[] getNext(String ps) {

    char[] p = ps.toCharArray();
    int[] next = new int[p.length];

    next[0] = -1; // 应该是 i 后移
    int j = 0;
    int k = -1;

    while (j < p.length - 1) {
      if (k == -1 || p[j] == p[k]) {
        // next[++j] = ++k;
        if (p[++j] == p[++k]) { // 当两个字符相等时要跳过
          next[j] = next[k];
        } else {
          next[j] = k;
        }
      } else {
        k = next[k];
      }
    }

    return next;

  }

  /**
   * KMP 算法
   *
   * @param ts 主串
   * @param ps 模式串
   * @return 如果找到，返回在主串中第一个字符出现的下标，否则返回 -1
   */
  public static int kmp(String ts, String ps) {

    char[] t = ts.toCharArray();
    char[] p = ps.toCharArray();

    int i = 0; // 主串的位置
    int j = 0; // 模式串的位置
    int[] next = getNext(ps);

    while (i < t.length && j < p.length) {
      if (j == -1 || t[i] == p[j]) { // 当 j 为 -1 时，需要移动 i，j 也要归 0
        i++;
        j++;
      } else {
        // i 不需要回溯了
        // i = i - j + 1;
        j = next[j]; // j 回到指定位置
      }
    }

    if (j == p.length) {
      return i - j;
    } else {
      return -1;
    }
  }
}
