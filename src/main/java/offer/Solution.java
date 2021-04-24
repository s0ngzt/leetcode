package offer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

class Solution {

  // 03 数组中重复的数字
  public int findRepeatNumber(int[] nums) {
    Set<Integer> set = new HashSet<>();
    int theNumber = -1;
    for (int num : nums) {
      if (!set.add(num)) {
        theNumber = num;
        break;
      }
    }
    return theNumber;
  }

  // 04 二维数组的查找
  // 从右上角 向左、向下找
  public boolean findNumberIn2DArray(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
      return false;
    }
    int h = matrix.length, w = matrix[0].length;
    int row = 0, column = w - 1;
    while (row < h && column >= 0) {
      int num = matrix[row][column];
      if (num == target) {
        return true;
      } else if (num > target) {
        column--;
      } else {
        row++;
      }
    }
    return false;
  }

  // 05 替换空格
  public String replaceSpace(String s) {
    int length = s.length();
    char[] array = new char[length * 3];
    int size = 0;
    for (int i = 0; i < length; i++) {
      char c = s.charAt(i);
      if (c == ' ') {
        array[size++] = '%';
        array[size++] = '2';
        array[size++] = '0';
      } else {
        array[size++] = c;
      }
    }
    return new String(array, 0, size);
  }

  // 06 从尾到头打印列表
  public int[] reversePrint(ListNode head) {

    Stack<Integer> stk = new Stack<>();
    ListNode temp = head;
    while (temp != null) {
      stk.push(temp.val);
      temp = temp.next;
    }
    int size = stk.size();
    int[] arr = new int[size];
    for (int i = 0; i < size; i++) {
      arr[i] = stk.pop();
    }
    return arr;
  }

  // 11 旋转数组的最小数字
  public int minArray(int[] numbers) {
    int left = 0, right = numbers.length - 1, middle;
    while (left < right) {
      middle = left + (right - left) / 2;
      if (numbers[middle] < numbers[right]) {
        right = middle;
      } else if (numbers[middle] == numbers[right]) {
        right -= 1;
      } else {
        left = middle + 1;
      }
    }
    return numbers[left];
  }

  // 12 矩阵中的路径
  public boolean exist(char[][] board, String word) {
    // TODO
    return false;
  }

  // 15
  public int hammingWeight(int n) {
    int count = 0;
    while (n != 0) {
      n = n & (n - 1);
      count++;
    }
    return count;
  }

  // 27
  public TreeNode mirrorTree(TreeNode root) {
    if (root == null) {
      return null;
    }
    var newHead = new TreeNode(root.val);
    if (root.left != null) {
      newHead.right = mirrorTree(root.left);
    }
    if (root.right != null) {
      newHead.left = mirrorTree(root.right);
    }
    return newHead;
  }

  // 28
  public boolean isSymmetric(TreeNode root) {
    return isMirror(root, root);
  }

  boolean isMirror(TreeNode t1, TreeNode t2) {
    if (t1 == null && t2 == null) {
      return true;
    }
    if (t1 == null || t2 == null) {
      return false;
    }
    return (t1.val == t2.val) && isMirror(t1.left, t2.right) && isMirror(t1.left, t2.right);
  }

  // 29 顺时针打印矩阵
  public int[] spiralOrder(int[][] matrix) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
      return new int[0];
    }
    int rows = matrix.length, cols = matrix[0].length;
    int[] order = new int[rows * cols];
    int index = 0;
    int left = 0, right = cols - 1, top = 0, bottom = rows - 1;
    while (left <= right && top <= bottom) {
      for (int column = left; column <= right; column++) {
        order[index++] = matrix[top][column];
      }
      for (int row = top + 1; row <= bottom; row++) {
        order[index++] = matrix[row][right];
      }
      if (left < right && top < bottom) {
        for (int column = right - 1; column > left; column--) {
          order[index++] = matrix[bottom][column];
        }
        for (int row = bottom; row > top; row--) {
          order[index++] = matrix[row][left];
        }
      }
      left++;
      right--;
      top++;
      bottom--;
    }
    return order;
  }


  // 50
  public char firstUniqChar(String s) {
    Map<Character, Integer> frequency = new HashMap<>();
    for (int i = 0; i < s.length(); i++) {
      char ch = s.charAt(i);
      frequency.put(ch, frequency.getOrDefault(ch, 0) + 1);
    }
    for (int i = 0; i < s.length(); ++i) {
      if (frequency.get(s.charAt(i)) == 1) {
        return s.charAt(i);
      }
    }
    return ' ';
  }

  // 55 - I
  public int maxDepth(TreeNode root) {
    if (root == null) {
      return 0;
    }
    if (root.left == null && root.right == null) {
      return 1;
    }
    if (root.left == null) {
      return 1 + maxDepth(root.right);
    }
    if (root.right == null) {
      return 1 + maxDepth(root.left);
    }
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
  }

  // 55 - II
  public boolean isBalanced(TreeNode root) {
    return height(root) >= 0;
  }

  public int height(TreeNode root) {
    if (root == null) {
      return 0;
    }
    int leftHeight = height(root.left);
    int rightHeight = height(root.right);
    if (leftHeight == -1 || rightHeight == -1 || Math.abs(leftHeight - rightHeight) > 1) {
      return -1;
    } else {
      return Math.max(leftHeight, rightHeight) + 1;
    }
  }

  // 56 - I 数组中数字出现的次数
  public int[] singleNumbers(int[] nums) {
    int ansAfterXor = 0;
    for (var num : nums) {
      ansAfterXor ^= num;
    }
    int div = 1;
    while ((div & ansAfterXor) == 0) {
      div <<= 1;
    }
    int a = 0, b = 0;
    for (int n : nums) {
      if ((div & n) != 0) {
        a ^= n;
      } else {
        b ^= n;
      }
    }
    return new int[]{a, b};

  }

  // 57
  public int[] twoSum(int[] nums, int target) {
    int l = 0, r = nums.length - 1;
    while (l < r) {
      int s = nums[l] + nums[r];
      if (s < target) {
        l++;
      } else if (s > target) {
        r--;
      } else {
        return new int[]{nums[l], nums[r]};
      }
    }
    return new int[0];
  }

}
