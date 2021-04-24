package leetcode;

import java.util.LinkedList;
import java.util.Stack;
import java.util.TreeSet;

public class SolutionHard {

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
