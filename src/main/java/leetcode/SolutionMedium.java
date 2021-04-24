package leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class SolutionMedium {

  // 264
  public int nthUglyNumber(int n) {
    int[] dp = new int[n + 1];
    dp[1] = 1;
    int p2 = 1, p3 = 1, p5 = 1;
    for (int i = 2; i <= n; i++) {
      int num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
      dp[i] = Math.min(Math.min(num2, num3), num5);
      if (dp[i] == num2) {
        p2++;
      }
      if (dp[i] == num3) {
        p3++;
      }
      if (dp[i] == num5) {
        p5++;
      }
    }
    return dp[n];
  }

  // 187
  public List<String> findRepeatedDnaSequences(String s) {
    List<String> res = new LinkedList<>();
    if (s.length() < 10) {
      return res;
    }
    Map<String, Integer> count = new HashMap<>();
    for (int i = 0; i <= s.length() - 10; i++) {
      var part = s.substring(i, i + 10);
      var times = count.getOrDefault(part, 0);
      if (times == 0) {
        count.put(part, 1);
      } else if (times == 1) {
        count.put(part, 2);
        res.add(part);
      }
    }
    return res;
  }

  // 201
  public int rangeBitwiseAnd(int left, int right) {
    while (right > left) {
      right &= (right - 1);
    }
    return right;
  }

  // 1686
  public int stoneGameVI(int[] aliceValues, int[] bobValues) {
    var stones = aliceValues.length;
    var totalBob = Arrays.stream(bobValues).sum();
    var addValues = new Integer[stones];
    for (int i = 0; i < stones; i++) {
      addValues[i] = aliceValues[i] + bobValues[i];
    }
    Arrays.sort(addValues, Comparator.comparingInt(a -> (int) a).reversed());
    var totalAlice = 0;
    for (int i = 0; i < stones; i += 2) {
      totalAlice += addValues[i];
    }
    return Integer.compare(totalAlice, totalBob);
  }

  // 1541
  public int minInsertions(String s) {
    int insertions = 0;
    int leftBracket = 0;
    int length = s.length();
    int index = 0;
    while (index < length) {
      char c = s.charAt(index);
      if (c == '(') {
        leftBracket++;
        index++;
      } else {  // ")"
        if (leftBracket > 0) {
          leftBracket--;
        } else {
          insertions++;
        }
        if (index < length - 1 && s.charAt(index + 1) == ')') {
          index += 2;
        } else {
          insertions++;
          index++;
        }
      }
    }
    insertions += leftBracket * 2;
    return insertions;
  }

  // 646
  public int findLongestChain(int[][] pairs) {
    Arrays.sort(pairs, Comparator.comparingInt(a -> a[0]));
    int N = pairs.length;
    int[] dp = new int[N];
    Arrays.fill(dp, 1);

    for (int j = 1; j < N; ++j) {
      for (int i = 0; i < j; ++i) {
        if (pairs[i][1] < pairs[j][0]) {
          dp[j] = Math.max(dp[j], dp[i] + 1);
        }
      }
    }

    int ans = 0;
    for (int x : dp) {
      if (x > ans) {
        ans = x;
      }
    }
    return ans;
  }

  // 516
  public int longestPalindromeSubSequence(String s) {
    int n = s.length();
    // 状态表示数组，两个维度分别表示区间的左右端点，数值表示区间最大回文长度
    int[][] f = new int[n][n];
    int res = 0;
    // 先枚举区间长度
    for (int len = 1; len <= n; len++) {
      // 再枚举区间左右端点
      for (int i = 0; i + len - 1 < n; i++) {
        int j = i + len - 1;
        // 特判长度为1的情况
        if (len == 1) {
          f[i][j] = 1;
        } else {
          // 不用当前两个端点为回文边界的情况
          f[i][j] = Math.max(f[i + 1][j], f[i][j - 1]);
          // 如果两个当前区间的两个端点相同，那就用
          if (s.charAt(i) == s.charAt(j)) {
            f[i][j] = f[i + 1][j - 1] + 2;
          }
        }
        // 取最大值
        res = Math.max(f[i][j], res);
      }
    }
    return res;
  }

  // 377
  public int combinationSum4(int[] nums, int target) {
    int[] dp = new int[target + 1];
    dp[0] = 1;
    for (int i = 1; i <= target; i++) {
      for (int num : nums) {
        if (num <= i) {
          dp[i] += dp[i - num];
        }
      }
    }
    return dp[target];
  }

  // 368
  public List<Integer> largestDivisibleSubset(int[] nums) {
    int len = nums.length;
    Arrays.sort(nums);

    int[] dp = new int[len];
    Arrays.fill(dp, 1);
    int maxSize = 1;
    int maxVal = dp[0];
    for (int i = 1; i < len; i++) {
      for (int j = 0; j < i; j++) {
        // 「没有重复元素」
        if (nums[i] % nums[j] == 0) {
          dp[i] = Math.max(dp[i], dp[j] + 1);
        }
      }

      if (dp[i] > maxSize) {
        maxSize = dp[i];
        maxVal = nums[i];
      }
    }

    List<Integer> res = new ArrayList<>();
    if (maxSize == 1) {
      res.add(nums[0]);
      return res;
    }

    for (int i = len - 1; i >= 0 && maxSize > 0; i--) {
      if (dp[i] == maxSize && maxVal % nums[i] == 0) {
        res.add(nums[i]);
        maxVal = nums[i];
        maxSize--;
      }
    }
    return res;
  }

  // 300
  public int lengthOfLIS(int[] nums) {
    if (nums.length == 0) {
      return 0;
    }
    int[] dp = new int[nums.length];
    dp[0] = 1;
    int maxAns = 1;
    for (int i = 1; i < nums.length; i++) {
      dp[i] = 1;
      for (int j = 0; j < i; j++) {
        if (nums[i] > nums[j]) {
          dp[i] = Math.max(dp[i], dp[j] + 1);
        }
      }
      maxAns = Math.max(maxAns, dp[i]);
    }
    return maxAns;
  }

  // 289
  public void gameOfLife(int[][] board) {

  }

  // 220
  public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
    int n = nums.length;
    Map<Long, Long> map = new HashMap<>();
    long w = (long) t + 1; // 桶的大小
    for (int i = 0; i < n; i++) {
      long id = getID(nums[i], w);
      // 同一个桶
      if (map.containsKey(id)) {
        return true;
      }
      // 相邻桶
      if (map.containsKey(id - 1) && Math.abs(nums[i] - map.get(id - 1)) < w) {
        return true;
      }
      if (map.containsKey(id + 1) && Math.abs(nums[i] - map.get(id + 1)) < w) {
        return true;
      }
      map.put(id, (long) nums[i]);
      if (i >= k) {
        map.remove(getID(nums[i - k], w));
      }
    }
    return false;
  }

  public long getID(long x, long w) {
    if (x >= 0) {
      return x / w;
    }
    return (x + 1) / w - 1;
  }

  // 213
  public int rob2(int[] nums) {
    int families = nums.length;
    if (families == 1) {
      return nums[0];
    } else if (families == 2) {
      return Math.max(nums[0], nums[1]);
    }
    return Math.max(robRange(nums, 0, families - 2), robRange(nums, 1, families - 1));
  }

  public int robRange(int[] nums, int start, int end) {
    int first = nums[start], second = Math.max(nums[start], nums[start + 1]);
    for (int i = start + 2; i <= end; i++) {
      int temp = second;
      second = Math.max(first + nums[i], second);
      first = temp;
    }
    return second;
  }

  // 198
  public int rob(int[] nums) {
    if (nums == null || nums.length == 0) {
      return 0;
    }
    int num = nums.length;
    if (num == 1) {
      return nums[0];
    }
    int[] dp = new int[num];
    dp[0] = nums[0];
    dp[1] = Math.max(nums[0], nums[1]);
    for (int i = 2; i < num; i++) {
      dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
    }
    return dp[num - 1];
  }

  // 91
  public int numDecoding(String s) {
    int len = s.length();
    if (len == 0) {
      return 0;
    }
    int[] dp = new int[len + 1];
    dp[0] = 1;
    if (s.charAt(0) == '0') {
      dp[1] = 0;
    } else {
      dp[1] = 1;
    }
    for (int i = 2; i <= len; i++) {
      var lastNum = s.charAt(i - 1) - '0';
      if (lastNum >= 1 && lastNum <= 9) {
        dp[i] += dp[i - 1];
      }
      lastNum = (s.charAt(i - 2) - '0') * 10 + s.charAt(i - 1) - '0';
      if (lastNum >= 10 && lastNum <= 26) {
        dp[i] += dp[i - 2];
      }
    }
    return dp[len];
  }

  // 73 矩阵置零
  public void setZeroes(int[][] matrix) {
    if (matrix.length == 0 || matrix[0].length == 0) {
      return;
    }
    boolean isFirstRowExistZero = false, isFirstColExistZero = false;
    for (int[] integers : matrix) {
      if (integers[0] == 0) {
        isFirstColExistZero = true;
        break;
      }
    }
    for (int j = 0; j < matrix[0].length; j++) {
      if (matrix[0][j] == 0) {
        isFirstRowExistZero = true;
        break;
      }
    }
    for (int i = 1; i < matrix.length; i++) {
      for (int j = 1; j < matrix[0].length; j++) {
        if (matrix[i][j] == 0) {
          matrix[i][0] = 0;
          matrix[0][j] = 0;
        }
      }
    }
    for (int i = 1; i < matrix.length; i++) {
      if (matrix[i][0] == 0) {
        for (int j = 1; j < matrix[0].length; j++) {
          matrix[i][j] = 0;
        }
      }
    }
    for (int j = 1; j < matrix[0].length; j++) {
      if (matrix[0][j] == 0) {
        for (int i = 1; i < matrix.length; i++) {
          matrix[i][j] = 0;
        }
      }
    }
    if (isFirstRowExistZero) {
      Arrays.fill(matrix[0], 0);
    }
    if (isFirstColExistZero) {
      for (int i = 0; i < matrix.length; i++) {
        matrix[i][0] = 0;
      }
    }
  }

  // 62 数学题
  public int uniquePaths(int m, int n) {
    long ans = 1;
    for (int x = n, y = 1; y < m; ++x, ++y) {
      ans = ans * x / y;
    }
    return (int) ans;
  }

}
