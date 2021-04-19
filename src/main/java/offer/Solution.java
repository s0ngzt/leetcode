package offer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

class Solution {

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
}
