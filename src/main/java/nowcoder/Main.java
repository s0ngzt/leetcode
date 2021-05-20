package nowcoder;

import java.util.Scanner;

class Main {

  public static int getMinCombination(int x) {
    int ans = 0, max = 9, level = 1;
    if (x > 45) {
      return -1;
    } else {
      while (x > 0) {
        if (x > max) {
          ans += max * level;
          level *= 10;
        } else {
          ans += x * level;
        }
        x -= max;
        max--;
      }
    }
    return ans;
  }

  public static int getMinCost(int len, String a, String b) {
    int[] ch1 = new int[26];
    int[] ch2 = new int[26];
    for (int i = 0; i < len; i++) {
      ch1[a.charAt(i) - 'a']++;
      ch2[b.charAt(i) - 'a']++;
    }
    int i = 0, j = 0, count = 0;
    while (i < 26 && j < 26) {
      if (ch1[i] > ch2[j]) {
        count += ch2[j] * Math.abs(i - j);
        ch1[i] -= ch2[j];
        do {
          j++;
        } while (j < 26 && ch2[j] == 0);
      } else if (ch1[i] < ch2[j]) {
        count += ch1[i] * Math.abs(i - j);
        ch2[j] -= ch1[i];
        do {
          i++;
        } while (i < 26 && ch1[i] == 0);
      } else {
        count += ch1[i] * Math.abs(i - j);
        do {
          j++;
        } while (j < 26 && ch2[j] == 0);
        do {
          i++;
        } while (i < 26 && ch1[i] == 0);
      }
    }
    return count;
  }

  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int x = sc.nextInt();
    sc.nextLine();
    var a = sc.nextLine();
    var b = sc.nextLine();
    System.out.println(getMinCost(x, a, b));
  }
}
