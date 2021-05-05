package leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

public class SolutionMedium {

  // 740
  public int deleteAndEarn(int[] nums) {
    if (nums == null || nums.length == 0) {
      return 0;
    }
    if (nums.length == 1) {
      return nums[0];
    }
    int len = nums.length;
    int max = nums[0];
    for (int i = 1; i < len; i++) {
      max = Math.max(max, nums[i]);
    }
    // new 一个新数组 <count>
    int[] count = new int[max + 1];
    for (int item : nums) {
      count[item]++;
    }
    int[] dp = new int[max + 1];
    dp[1] = count[1]; // count[1] * 1
    dp[2] = Math.max(dp[1], count[2] * 2);
    // 动态规划求解
    for (int i = 3; i <= max; ++i) {
      dp[i] = Math.max(dp[i - 1], dp[i - 2] + i * count[i]);
    }
    return dp[max];
  }

  // 554 砖墙
  public int leastBricks(List<List<Integer>> wall) {
    int n = wall.size();
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0, sum = 0; i < n; i++, sum = 0) {
      for (int cur : wall.get(i)) {
        sum += cur;
        map.put(sum, map.getOrDefault(sum, 0) + 1);
      }
      map.remove(sum);
    }
    int ans = n;
    for (int u : map.keySet()) {
      int cnt = map.get(u);
      ans = Math.min(ans, n - cnt);
    }
    return ans;
  }

  // 47 全排列 II
  public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> perm = new ArrayList<>();
    boolean[] visited = new boolean[nums.length];
    Arrays.sort(nums);
    backtrack(nums, visited, perm, res);
    return res;
  }

  private void backtrack(int[] nums, boolean[] visited, List<Integer> perm,
      List<List<Integer>> res) {
    if (perm.size() == nums.length) {
      res.add(new ArrayList<>(perm));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      // 已经添加过的元素 或 上一个元素和当前相同，且没有访问过
      if (visited[i] || (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])) {
        continue;
      }
      perm.add(nums[i]);
      visited[i] = true;
      backtrack(nums, visited, perm, res);
      visited[i] = false;
      perm.remove(perm.size() - 1);
    }
  }

  // 3 无重复字符的最长子串
  public int lengthOfLongestSubstring(String s) {
    Set<Character> set = new HashSet<>();
    int n = s.length();
    int right = -1, ans = 0;
    for (int i = 0; i < n; ++i) {
      if (i != 0) {
        set.remove(s.charAt(i - 1));
      }
      while (right + 1 < n && !set.contains(s.charAt(right + 1))) {
        set.add(s.charAt(right + 1));
        ++right;
      }
      ans = Math.max(ans, right - i + 1);
    }
    return ans;
  }

  // 438 找到字符串中所有字母异位词
  // 给定一个字符串 s 和一个非空字符串 p，
  // 找到 s 中所有是 p 的字母异位词的子串，
  // 返回这些子串的起始索引。
  public List<Integer> findAnagrams(String s, String p) {
    int[] freq = new int[26];

    // 表示窗口内相差的字符的数量
    int diff = 0;
    // freq 统计频数
    for (char c : p.toCharArray()) {
      freq[c - 'a']++;
      diff++;
    }

    int left = 0, right = 0, length = s.length();
    char[] array = s.toCharArray();
    List<Integer> result = new ArrayList<>();

    while (right < length) {
      char rightChar = array[right];

      // 是 p 中的字符
      if (freq[rightChar - 'a'] > 0) {
        freq[rightChar - 'a']--;
        // 差距减少
        diff--;
        right++;

        //差距减少为0时 说明窗口内为所求
        if (diff == 0) {
          result.add(left);
        }
      } else {
        // 两种情况
        // 1. rightChar 是 p 以外的字符串如"c" "abc" "ab"
        //    此时 left 和 right 都应该指向 c后面的位置
        // 2. rightChar 是 p 内的字符串 但是是额外的一个 char 如第二个"b" 例 "abb" "ab"
        //    此时 right 不变 left 应该指向第一个 b 后面的位置
        // 对于第一种情况，left 和 right 都应该定位到 c，所以要恢复 freq 和 diff
        // 对于第二种情况 此时 freq[array[right]-'a']=0
        // 让 left 移动到第一个 b 后面的位置 这样就融入了新的 b（第二个b）
        while (freq[array[right] - 'a'] <= 0 && left < right) {
          freq[array[left] - 'a']++;
          left++;
          diff++;
        }

        if (left == right) {
          if (freq[array[left] - 'a'] <= 0) {
            //用来处理第一种情况 移动到这个字符后面的位置
            left++;
            right++;
          }
        }
      }
    }

    return result;
  }

  // 567 字符串的排列
  public boolean checkInclusion(String s1, String s2) {
    int n = s1.length(), m = s2.length();
    if (n > m) {
      return false;
    }
    int[] cnt1 = new int[26];
    int[] cnt2 = new int[26];
    for (int i = 0; i < n; ++i) {
      ++cnt1[s1.charAt(i) - 'a'];
      ++cnt2[s2.charAt(i) - 'a'];
    }
    if (Arrays.equals(cnt1, cnt2)) {
      return true;
    }
    for (int i = n; i < m; ++i) {
      // 滑动
      ++cnt2[s2.charAt(i) - 'a'];
      --cnt2[s2.charAt(i - n) - 'a'];
      if (Arrays.equals(cnt1, cnt2)) {
        return true;
      }
    }
    return false;
  }

  // 322 零钱兑换
  public int coinChange(int[] coins, int amount) {
    // 状态 dp[i] 表示金额为 i 时，组成的最小硬币个数
    // 推导 dp[i]  = min(dp[i-1], dp[i-2], dp[i-5]) + 1, 前提 i-coins[j] > 0
    // 初始化为最大值 dp[i]=amount+1
    // 返回值 dp[n] or dp[n]>amount =>-1
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
      for (int coin : coins) {
        if (i - coin >= 0) {
          dp[i] = Math.min(dp[i], dp[i - coin] + 1);
        }
      }
    }
    return (dp[amount] > amount) ? -1 : dp[amount];
  }

  // 1143
  public int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
      char c1 = text1.charAt(i - 1);
      for (int j = 1; j <= n; j++) {
        char c2 = text2.charAt(j - 1);
        if (c1 == c2) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }
    return dp[m][n];
  }

  // 139 单词拆分
  public boolean wordBreak(String s, List<String> wordDict) {
    // f[i] 表示前i个字符是否可以被切分
    // f[i] = f[j] && s[j+1~i] in wordDict
    // f[0] = true
    // return f[len]
    Set<String> wordDictSet = new HashSet<>(wordDict);
    int m = s.length();
    boolean[] dp = new boolean[m + 1];
    dp[0] = true;
    for (int i = 1; i <= m; i++) {
      for (int j = 0; j < i; j++) {
        if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
          dp[i] = true;
          break;
        }
      }
    }
    return dp[m];
  }


  // 45 跳跃游戏 II
  // 给定一个非负整数数组，你最初位于数组的第一个位置。
  // 数组中的每个元素代表你在该位置可以跳跃的最大长度。
  // 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
  // 假设你总是可以到达数组的最后一个位置。
  public int jump(int[] nums) {
    // 状态：f[i] 表示从起点到当前位置最小次数
    // 推导：f[i] = f[j],a[j]+j >=i, min(f[j]+1)
    // 初始化：f[0] = 0
    // 结果：f[n-1]
    int m = nums.length;
    int[] dp = new int[m];
    dp[0] = 0;
    for (int i = 1; i < m; i++) {
      // f[i] 最大值为i
      dp[i] = i;
      // 遍历之前结果取一个最小值+1
      for (int j = 0; j < i; j++) {
        if (nums[j] + j >= i) {
          dp[i] = Math.min(dp[j] + 1, dp[i]);
        }
      }
    }
    return dp[m - 1];
  }

  // 动态规划 + 贪心优化
  public int jumpV2(int[] nums) {
    int m = nums.length;
    int[] dp = new int[m];
    dp[0] = 0;
    for (int i = 1; i < m; i++) {
      // 取第一个能跳到当前位置的点即可
      // 因为跳跃次数的结果集是单调递增的，所以贪心思路是正确的
      int index = 0;
      while (index < m && index + nums[index] < i) {
        index++;
      }
      dp[i] = dp[index] = 1;
    }
    return dp[m - 1];
  }

  // 55 跳跃游戏
  // 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
  // 数组中的每个元素代表你在该位置可以跳跃的最大长度。
  // 判断你是否能够到达最后一个下标。
  public boolean canJump(int[] nums) {
    // 思路   看最后一跳
    // 状态   f[i] 表示是否能从0跳到i
    // 推导   f[i] = OR(f[j],j<i&&j能跳到i) 判断之前所有的点最后一跳是否能跳到当前点
    // 初始化 f[0] = 0
    // 结果   f[n-1]
    int m = nums.length;
    if (m == 0) {
      return true;
    }
    boolean[] dp = new boolean[m];
    dp[0] = true;
    for (int i = 1; i < m; i++) {
      for (int j = 0; j < i; j++) {
        if (dp[j] && nums[j] + j >= i) {
          dp[i] = true;
          break;
        }
      }
    }
    return dp[m - 1];
  }

  // 633 平方数之和
  public boolean judgeSquareSum(int c) {
    int left = 0;
    int right = (int) Math.sqrt(c);
    while (left <= right) {
      int sum = left * left + right * right;
      if (sum == c) {
        return true;
      } else if (sum > c) {
        right--;
      } else {
        left++;
      }
    }
    return false;
  }

  // 63 不同路径 II
  // 网格中的“障碍物”和“空位置”分别用 1 和 0 表示。
  public int uniquePathsWithObstacles(int[][] obstacleGrid) {

    if (obstacleGrid[0][0] == 1) {
      return 0;
    }

    int m = obstacleGrid.length, n = obstacleGrid[0].length;
    // dp[i][j] 表示 i,j 到 0,0 路径数
    int[][] dp = new int[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        dp[i][j] = 1;
      }
    }
    // 处理第 0 行
    for (int i = 1; i < m; i++) {
      if (obstacleGrid[i][0] == 1 || dp[i - 1][0] == 0) {
        dp[i][0] = 0;
      }
    }
    // 处理第 0 列
    for (int j = 1; j < n; j++) {
      if (obstacleGrid[0][j] == 1 || dp[0][j - 1] == 0) {
        dp[0][j] = 0;
      }
    }

    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        if (obstacleGrid[i][j] == 1) {
          dp[i][j] = 0;
        } else {
          dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
      }
    }
    return dp[m - 1][n - 1];
  }

  // 64 最小路径和
  public int minPathSum(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    if (m == 0 || n == 0) {
      return 0;
    }
    // 复用原来的矩阵列表
    // 初始化：f[i][0]、f[0][j]
    for (int i = 1; i < m; i++) {
      grid[i][0] = grid[i][0] + grid[i - 1][0];
    }
    for (int j = 1; j < n; j++) {
      grid[0][j] = grid[0][j] + grid[0][j - 1];
    }
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        grid[i][j] = Math.min(grid[i][j - 1], grid[i - 1][j]) + grid[i][j];
      }
    }
    return grid[m - 1][n - 1];
  }

  // 120
  public int minimumTotal(List<List<Integer>> triangle) {
    var l = triangle.size();
    var dp = new int[l][l];

    for (int i = 0; i < l; i++) {
      for (int j = 0; j < i + 1; j++) {
        dp[i][j] = triangle.get(i).get(j);
      }
    }
    // 自底向上
    for (int i = l - 2; i >= 0; i--) {
      for (int j = 0; j < i + 1; j++) {
        dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle.get(i).get(j);
      }
    }
    // 4、答案
    return dp[0][0];
  }

  // 260
  public int[] singleNumberIII(int[] nums) {
    int diff = 0;
    for (int num : nums) {
      diff ^= num;
    }
    var result = new int[2];
    // 去掉末尾的 1 后异或 diff 就得到最后一个 1 的位置
    diff = (diff & (diff - 1)) ^ diff;
    for (int num : nums) {
      if ((num & diff) == 0) {
        result[0] ^= num;
      } else {
        result[1] ^= num;
      }
    }
    return result;
  }

  // 137
  public int singleNumberII(int[] nums) {
    // 统计每位 1 的个数
    int result = 0;
    for (int i = 0; i < 32; i++) {
      int sum = 0;
      for (int num : nums) {
        // 统计1的个数
        sum += (num >> i) & 1;
      }
      // 还原位 00^10=10，用 '|' 也可以
      result ^= (sum % 3) << i;
    }
    return result;
  }

  // 200 岛屿数量
  public int numIslands(char[][] grid) {
    var count = 0;
    int l = grid.length, w = grid[0].length;
    for (int i = 0; i < l; i++) {
      for (int j = 0; j < w; j++) {
        if (grid[i][j] == '1' && dfs(grid, i, j) >= 1) {
          count++;
        }
      }
    }
    return count;
  }

  private int dfs(char[][] grid, int i, int j) {
    if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length) {
      return 0;
    }
    if (grid[i][j] == '1') {
      grid[i][j] = 'x';
      return dfs(grid, i - 1, j) + dfs(grid, i, j - 1) + dfs(grid, i + 1, j) + dfs(grid, i, j + 1)
          + 1;
    }
    return 0;
  }

  // 1689 十-二进制数的最少数目
  public int minPartitions(String n) {
    int min = 0;
    for (var c : n.toCharArray()) {
      if (c - '0' > min) {
        min = c - '0';
      }
    }
    return min;
  }

  // 394 字符串编码
  public String decodeString(String s) {
    if (s.length() == 0) {
      return "";
    }
    Deque<Character> stack = new LinkedList<>();
    for (var chr : s.toCharArray()) {
      if (chr == ']') {
        Deque<Character> temp = new LinkedList<>();
        while (!stack.isEmpty() && stack.peekLast() != '[') {
          temp.offerFirst(stack.pollLast());
        }
        stack.pollLast();  // pop '['
        // 拿数字
        int times = 0, pow = 1;
        while (!stack.isEmpty() && Character.isDigit(stack.peekLast())) {
          assert (!stack.isEmpty());
          times += pow * (stack.pollLast() - '0');
          pow *= 10;
        }

        for (int i = 0; i < times; i++) {
          for (var c : temp) {
            stack.offerLast(c);
          }
        }
      } else {
        stack.offerLast(chr);
      }
    }

    var sb = new StringBuilder();
    for (var c : stack) {
      sb.append(c);
    }
    return sb.toString();
  }

  // 150 逆波兰表达式求值
  public int evalRPN(String[] tokens) {
    var s = new Stack<Integer>();
    for (String token : tokens) {
      switch (token) {
        case "+": {
          var a1 = s.pop();
          var a2 = s.pop();
          s.push(a2 + a1);
          break;
        }
        case "-": {
          var a1 = s.pop();
          var a2 = s.pop();
          s.push(a2 - a1);
          break;
        }
        case "*": {
          var a1 = s.pop();
          var a2 = s.pop();
          s.push(a2 * a1);
          break;
        }
        case "/": {
          var a1 = s.pop();
          var a2 = s.pop();
          s.push(a2 / a1);
          break;
        }
        default:
          s.push(Integer.parseInt(token));
          break;
      }
    }
    return s.peek();
  }

  // 1011 & 410 (hard)
  public int shipWithinDays(int[] weights, int D) {
    int maxNum = 0, sum = 0;
    for (var num : weights) {
      sum += num;
      if (num > maxNum) {
        maxNum = num;
      }
    }
    if (D == 1) {
      return sum;
    }
    // 返回 和 的 最小值
    int low = maxNum, high = sum, mid;
    while (low < high) {
      mid = low + (high - low) / 2;
      if (calSum(mid, D, weights)) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    return low;
  }

  boolean calSum(int mid, int m, int[] nums) {
    int sum = 0, count = 0;
    for (var num : nums) {
      sum += num;
      if (sum > mid) {
        sum = num;
        count++;
        // 分成 m 块，只需要插桩 m -1 个
        if (count > m - 1) {
          return false;
        }
      }
    }
    return true;
  }

  // 807
  public int maxIncreaseKeepingSkyline(int[][] grid) {
    int l = grid.length, w = grid[0].length;
    int[] rowMax = new int[l];
    int[] colMax = new int[w];

    for (int r = 0; r < l; ++r) {
      for (int c = 0; c < w; ++c) {
        rowMax[r] = Math.max(rowMax[r], grid[r][c]);
        colMax[c] = Math.max(colMax[c], grid[r][c]);
      }
    }

    int ans = 0;
    for (int r = 0; r < l; ++r) {
      for (int c = 0; c < w; ++c) {
        ans += Math.min(rowMax[r], colMax[c]) - grid[r][c];
      }
    }

    return ans;
  }

  // 1833
  public int maxIceCream(int[] costs, int coins) {
    int ans = 0;
    Arrays.sort(costs);

    for (var cost : costs) {
      if (coins >= cost) {
        ans++;
        coins -= cost;
      }
      if (coins <= 0) {
        break;
      }
    }
    return ans;
  }

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
