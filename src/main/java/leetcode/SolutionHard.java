package leetcode;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Stack;
import java.util.TreeSet;

public class SolutionHard {


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