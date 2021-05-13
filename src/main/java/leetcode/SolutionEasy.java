package leetcode;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

public class SolutionEasy {

  // 733 图像渲染
  public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
    int[][] directions = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    int m = image.length, n = image[0].length;
    boolean[][] seen = new boolean[m][n];
    Queue<int[]> queue = new LinkedList<>();
    queue.offer(new int[]{sr, sc});
    seen[sr][sc] = true;
    int oldColor = image[sr][sc];

    // BFS
    while (!queue.isEmpty()) {
      int[] cell = queue.poll();
      int i = cell[0], j = cell[1];
      image[i][j] = newColor;
      for (int d = 0; d < 4; ++d) {
        int ni = i + directions[d][0];
        int nj = j + directions[d][1];
        if (ni >= 0 && ni < m && nj >= 0 && nj < n && !seen[ni][nj] && image[ni][nj] == oldColor) {
          queue.offer(new int[]{ni, nj});
          seen[ni][nj] = true;
        }
      }
    }
    return image;
  }

  // 1486 数组异或操作
  public int xorOperation(int n, int start) {
    int b0 = n & start & 1;
    int s = start / 2;
    int res = computeXOR(s - 1) ^ computeXOR(s + n - 1);
    return (res << 1) + b0;

  }

  private int computeXOR(int n) {
    switch (n % 4) {
      case 0:
        return n;
      case 1:
        return 1;
      case 2:
        return n + 1;
    }
    return 0;
  }

  // 1365
  public int[] smallerNumbersThanCurrent(int[] nums) {
    int[] cnt = new int[101]; // 0 <= nums[i] <= 100
    int n = nums.length;
    for (int num : nums) {
      cnt[num]++;
    }
    for (int i = 1; i <= 100; i++) {
      cnt[i] += cnt[i - 1]; // 累加
    }
    int[] ret = new int[n];
    for (int i = 0; i < n; i++) {
      ret[i] = nums[i] == 0 ? 0 : cnt[nums[i] - 1];
    }
    return ret;
  }

  // 1720
  public int[] decode(int[] encoded, int first) {
    int n = encoded.length + 1;
    int[] origin = new int[n];
    origin[0] = first;
    for (int i = 1; i < n; i++) {
      origin[i] = origin[i - 1] ^ encoded[i - 1];
    }
    return origin;
  }

  // 690 员工的重要性
  public int getImportance(List<Employee> employees, int id) {
    Map<Integer, Employee> map = new HashMap<>();
    for (Employee employee : employees) {
      map.put(employee.id, employee);
    }
    int total = 0;
    Queue<Integer> queue = new LinkedList<>();
    queue.offer(id);
    while (!queue.isEmpty()) {
      int curId = queue.poll();
      Employee employee = map.get(curId);
      total += employee.importance;
      List<Integer> subordinates = employee.subordinates;
      for (int subId : subordinates) {
        queue.offer(subId);
      }
    }
    return total;
  }

  // 1678
  public String interpret(String command) {
    StringBuilder sb = new StringBuilder();
    var chars = command.toCharArray();
    int index = 0;
    while (index < chars.length) {
      if (chars[index] == 'G') {
        sb.append('G');
        index++;
      } else {
        if (chars[index + 1] == ')') {
          sb.append('o');
          index += 2;
        } else {
          sb.append("al");
          index += 4;
        }
      }
    }
    return sb.toString();
  }

  // 344 反转字符串
  public void reverseString(char[] s) {
    int n = s.length;
    for (int l = 0, r = n - 1; l < r; ++l, --r) {
      char tmp = s[l];
      s[l] = s[r];
      s[r] = tmp;
    }
  }

  // 70 爬楼梯
  public int climbStairs(int n) {
    // f[i] = f[i-1] + f[i-2]
    if (n == 1 || n == 0) {
      return n;
    }
    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) {
      dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
  }

  // 121
  public int maxProfit(int[] prices) {
    int minPrice = Integer.MAX_VALUE;
    int maxProfit = 0;
    for (int price : prices) {
      if (price < minPrice) {
        minPrice = price;
      } else if (price - minPrice > maxProfit) {
        maxProfit = price - minPrice;
      }
    }
    return maxProfit;
  }

  // 263
  public boolean isUgly(int num) {

    if (num <= 0) {
      return false;
    }
    int[] primes = {2, 3, 5};
    for (int prime : primes) {
      while (num % prime == 0) {
        num /= prime;
      }
    }
    return num == 1;

  }

  // 27
  public int removeElement(int[] nums, int val) {
    int l = 0, r = nums.length;
    while (l < r) {
      if (nums[l] == val) {
        nums[l] = nums[r - 1];
        r--;
      } else {
        l++;
      }
    }
    return l;
  }

  // 1576
  public String modifyString(String s) {
    var chars = s.toCharArray();
    int n = chars.length;
    for (int i = 0; i < chars.length; i++) {
      if (chars[i] == '?') {
        char left = i == 0 ? 'S' : chars[i - 1];
        char right = i == n - 1 ? 'S' : chars[i + 1];
        char temp = 'a';
        while (temp == left || temp == right) {
          temp++;
        }
        chars[i] = temp;
      }
    }
    return new String(chars);
  }

  // 1207
  public boolean uniqueOccurrences(int[] arr) {
    Map<Integer, Integer> times = new HashMap<>();
    for (int x : arr) {
      times.put(x, times.getOrDefault(x, 0) + 1);
    }
    Set<Integer> deduplication = new HashSet<>();
    for (Map.Entry<Integer, Integer> x : times.entrySet()) {
      if (!deduplication.add(x.getValue())) {
        return false;
      }
    }
    return true;
  }

  // 136
  public int singleNumber(int[] nums) {
    int ans = 0;
    for (var num : nums) {
      ans ^= num;
    }
    return ans;
  }

}
