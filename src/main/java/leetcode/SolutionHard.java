package leetcode;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Stack;
import java.util.TreeSet;

public class SolutionHard {

  // 1473
  public int minCost(int[] hs, int[][] cost, int m, int n, int t) {

    int INF = 0x3f3f3f3f;
    int[][][] f = new int[m + 1][n + 1][t + 1];

    // 不存在分区数量为 0 的状态
    for (int i = 0; i <= m; i++) {
      for (int j = 0; j <= n; j++) {
        f[i][j][0] = INF;
      }
    }

    for (int i = 1; i <= m; i++) {
      int color = hs[i - 1];
      for (int j = 1; j <= n; j++) {
        for (int k = 1; k <= t; k++) {
          // 形成分区数量大于房子数量，状态无效
          if (k > i) {
            f[i][j][k] = INF;
            continue;
          }

          // 第 i 间房间已经上色
          if (color != 0) {
            if (j == color) { // 只有与「本来的颜色」相同的状态才允许被转移
              int tmp = INF;
              // 先从所有「第 i 间房形成新分区」方案中选最优（即与上一房间颜色不同）
              for (int p = 1; p <= n; p++) {
                if (p != j) {
                  tmp = Math.min(tmp, f[i - 1][p][k - 1]);
                }
              }
              // 再结合「第 i 间房不形成新分区」方案中选最优（即与上一房间颜色相同）
              f[i][j][k] = Math.min(f[i - 1][j][k], tmp);

            } else { // 其余状态无效
              f[i][j][k] = INF;
            }

            // 第 i 间房间尚未上色
          } else {
            int u = cost[i - 1][j - 1];
            int tmp = INF;
            // 先从所有「第 i 间房形成新分区」方案中选最优（即与上一房间颜色不同）
            for (int p = 1; p <= n; p++) {
              if (p != j) {
                tmp = Math.min(tmp, f[i - 1][p][k - 1]);
              }
            }
            // 再结合「第 i 间房不形成新分区」方案中选最优（即与上一房间颜色相同）
            // 并将「上色成本」添加进去
            f[i][j][k] = Math.min(tmp, f[i - 1][j][k]) + u;
          }
        }
      }
    }

    // 从「考虑所有房间，并且形成分区数量为 t」的所有方案中找答案
    int ans = INF;
    for (int i = 1; i <= n; i++) {
      ans = Math.min(ans, f[m][i][t]);
    }
    return ans == INF ? -1 : ans;
  }

  // 403 青蛙过河
  public boolean canCross(int[] stones) {
    int numStone = stones.length;

    boolean[][] dp = new boolean[numStone][numStone];
    dp[0][0] = true;
    for (int i = 1; i < numStone; i++) {
      if (stones[i] - stones[i - 1] > i) {
        return false;
      }
    }

    for (int i = 1; i < numStone; i++) {
      for (int j = i - 1; j >= 0; j--) {
        int k = stones[i] - stones[j];
        if (k > j + 1) {
          break;
        }
        dp[i][k] = dp[j][k - 1] || dp[j][k] || dp[j][k + 1];
        if (i == numStone - 1 && dp[i][k]) {
          return true;
        }
      }
    }

    return false;
  }


  // 72 编辑距离
  public int minDistance(String word1, String word2) {
    // dp[i][j] 表示 a 字符串的前 i 个字符编辑为 b 字符串的前 j 个字符最少需要多少次操作
    // dp[i][j] = OR(dp[i-1][j-1]，a[i]==b[j],min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1)
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 0; i < m + 1; i++) {
      dp[i][0] = i;
    }
    for (int j = 1; j < n + 1; j++) {
      dp[0][j] = j;
    }
    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        // 相等则不需要操作
        if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
          dp[i][j] = dp[i - 1][j - 1];
        } else { // 否则取删除、插入、替换最小操作次数的值+1
          dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
        }
      }
    }
    return dp[m][n];
  }

  // 84 柱状图中的最大矩形
  // TODO https://github.com/greyireland/algorithm-pattern/blob/master/data_structure/stack_queue.md
  public int largestRectangleArea(int[] heights) {
    if (heights.length == 0) {
      return 0;
    }
    int numHeight = heights.length;
    Deque<Integer> stack = new LinkedList<>();
    int max = 0;
    for (int i = 0; i <= numHeight; i++) {
      int cur = i == numHeight ? 0 : heights[i];
      // 当前高度小于栈，则将栈内元素都弹出计算面积
      while (!stack.isEmpty() && cur <= heights[stack.peek()]) {
        var h = heights[stack.pop()];
        // 计算宽度
        var w = i;
        if (!stack.isEmpty()) {
          var peek = stack.peek();
          w = i - peek - 1;
        }
        var area = h * w;
        if (area > max) {
          max = area;
        }
      }
      // 记录索引即可获取对应元素
      stack.push(i);
    }
    return max;
  }

  // 1106
  public boolean parseBoolExpr(String expression) {
    // 双栈
    Stack<Character> ops = new Stack<>(); // 操作符
    Stack<Character> chs = new Stack<>(); // 操作数
    boolean tmp;

    for (var c : expression.toCharArray()) {

      if (c == '!' || c == '|' || c == '&') {
        ops.push(c);
      } else if (c == ')') {
        var stage = new LinkedList<Boolean>();

        while (chs.peek() != '(') {
          stage.add(chs.pop() == 't');
        }
        chs.pop(); // '(' 出栈

        switch (ops.pop()) {
          case '!':
            assert (!stage.isEmpty());
            chs.push(stage.peek() ? 'f' : 't');
            break;
          case '|':
            tmp = false;
            for (var b : stage) {
              tmp = tmp || b;
            }
            chs.push(tmp ? 't' : 'f');
            break;
          case '&':
            tmp = true;
            for (var b : stage) {
              tmp = tmp && b;
            }
            chs.push(tmp ? 't' : 'f');
            break;
        }
      } else if (c != ',') {
        chs.push(c);
      }
    }

    return chs.peek() == 't';
  }

  // 154
  // [3.3.1.3]
  public int findMinHard(int[] nums) {
    int l = 0, r = nums.length - 1;
    while (l < r) {
      int m = l + (r - l) / 2;
      if (nums[m] < nums[r]) {
        r = m;
      } else if (nums[m] == nums[r]) {
        r = r - 1;
      } else {
        l = m + 1;
      }
    }
    return nums[l];
  }

  // 363
  public int maxSumSubMatrix(int[][] matrix, int k) {
    int m = matrix.length, n = matrix[0].length;

    int[][] prefixSum = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        prefixSum[i][j] =
            prefixSum[i - 1][j] + prefixSum[i][j - 1] - prefixSum[i - 1][j - 1] + matrix[i - 1][j
                - 1];
      }
    }

    int ans = Integer.MIN_VALUE;
    for (int top = 1; top <= m; top++) {
      for (int bottom = top; bottom <= m; bottom++) {
        TreeSet<Integer> ts = new TreeSet<>();
        ts.add(0);
        for (int r = 1; r <= n; r++) {
          int right = prefixSum[bottom][r] - prefixSum[top - 1][r];
          Integer left = ts.ceiling(right - k);
          if (left != null) {
            int cur = right - left;
            ans = Math.max(ans, cur);
          }
          ts.add(right);
        }
      }
    }
    return ans;
  }
}
