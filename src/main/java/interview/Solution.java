package interview;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

class Solution {

  // 面试题 01.01. 判定字符是否唯一
  public boolean isUnique(String s) {
    if (s.length() > 26) {
      return false;
    }
    boolean[] mark = new boolean[26];
    for (var c : s.toCharArray()) {
      if (mark[c - 'a']) {
        return false;
      }
      mark[c - 'a'] = true;
    }
    return true;
  }

  // 面试题 01.02. 判定是否互为字符重排
  public boolean CheckPermutation(String s1, String s2) {
    if (s1.length() != s2.length()) {
      return false;
    }
    int[] count = new int[26];
    for (var c : s1.toCharArray()) {
      count[c - 'a']++;
    }
    for (var c : s2.toCharArray()) {
      count[c - 'a']--;
      if (count[c - 'a'] < 0) {
        return false;
      }
    }
    return true;
  }

  // 面试题 01.03. URL化
  public String replaceSpaces(String S, int length) {
    char[] res = new char[3 * length];
    int i = 0;
    for (int j = 0; j < length; j++) {
      if (S.charAt(j) == ' ') {
        res[i++] = '%';
        res[i++] = '2';
        res[i++] = '0';
      } else {
        res[i++] = S.charAt(j);
      }
    }
    return new String(res, 0, i);
  }

  // 面试题 01.04. 回文排列
  public boolean canPermutePalindrome(String s) {
    Set<Character> set = new HashSet<>();
    for (char ch : s.toCharArray()) {
      if (!set.add(ch)) {
        set.remove(ch);
      }
    }
    return set.size() <= 1;
  }

  // 面试题 01.05. 一次编辑
  public boolean oneEditAway(String first, String second) {
    int n1 = first.length(), n2 = second.length();
    int diff = n1 - n2;
    if (Math.abs(diff) > 1) {
      return false;
    }
    int op = 1;
    for (int i = 0, j = 0; i < n1 && j < n2; ++i, ++j) {
      boolean notSame = first.charAt(i) != second.charAt(j);
      if (notSame) {
        if (diff == 1) {
          --j;
        } else if (diff == -1) {
          --i;
        }
        --op;
      }
      if (op < 0) {
        return false;
      }
    }
    return true;
  }

  // 面试题 01.06. 字符串压缩
  public String compressString(String S) {
    StringBuilder sb = new StringBuilder();
    char[] cs = S.toCharArray();
    int slow = 0, fast = 0, count = 0, n = S.length();
    while (fast < n) {
      while (fast < n && cs[fast] == cs[slow]) {
        fast++;
      }
      sb.append(cs[slow]);
      sb.append((fast - slow));
      slow = fast;
    }
    var s1 = sb.toString();
    return s1.length() < S.length() ? s1 : S;
  }

  // 面试题 01.07. 旋转矩阵
  public void rotate(int[][] matrix) {
    int n = matrix.length;
    if (n < 2) {
      return;
    }
    for (int i = 0; i < n / 2; ++i) {
      for (int j = 0; j < n; ++j) {
        int temp = matrix[i][j];
        matrix[i][j] = matrix[n - i - 1][j];
        matrix[n - i - 1][j] = temp;
      }
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        int temp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = temp;
      }
    }
  }

  // 面试题 01.08. 零矩阵
  public void setZeroes(int[][] matrix) {
    int m = matrix.length, n = matrix[0].length;
    boolean flagCol0 = false;
    for (int i = 0; i < m; i++) {
      if (matrix[i][0] == 0) {
        flagCol0 = true;
      }
      for (int j = 1; j < n; j++) {
        if (matrix[i][j] == 0) {
          matrix[i][0] = matrix[0][j] = 0;
        }
      }
    }
    for (int i = m - 1; i >= 0; i--) {
      for (int j = 1; j < n; j++) {
        if (matrix[i][0] == 0 || matrix[0][j] == 0) {
          matrix[i][j] = 0;
        }
      }
      if (flagCol0) {
        matrix[i][0] = 0;
      }
    }
  }

  // 面试题 01.09. 字符串轮转
  public boolean isFlippedString(String s1, String s2) {
    return s1.length() == s2.length() && (s1 + s1).contains(s2);
  }

  // 面试题 02.01. 移除重复节点
  public ListNode removeDuplicateNodes(ListNode head) {
    ListNode ob = head;
    while (ob != null) {
      ListNode oc = ob;
      while (oc.next != null) {
        if (oc.next.val == ob.val) {
          oc.next = oc.next.next;
        } else {
          oc = oc.next;
        }
      }
      ob = ob.next;
    }
    return head;
  }

  // 面试题 02.02. 返回倒数第 k 个节点
  // 双指针
  public int kthToLast(ListNode head, int k) {
    ListNode dummyHead = new ListNode(-1);
    dummyHead.next = head;
    ListNode s = dummyHead;
    ListNode f = dummyHead;
    for (int i = 0; i < k; i++) {
      f = f.next;
    }
    while (f != null) {
      f = f.next;
      s = s.next;
    }
    return s.val;
  }

  // 面试题 02.03. 删除中间节点
  public void deleteNode(ListNode node) {
    node.val = node.next.val;
    node.next = node.next.next;

  }

  // 面试题 02.04. 分割链表
  // 想多了，双链表
  public ListNode partition(ListNode head, int x) {
    ListNode less = new ListNode(0);
    ListNode lessHead = less;
    ListNode greater = new ListNode(0);
    ListNode greaterHead = greater;
    while (head != null) {
      if (head.val < x) {
        less.next = head;
        less = less.next;
      } else {
        greater.next = head;
        greater = greater.next;
      }
      head = head.next;
    }
    greater.next = null;
    less.next = greaterHead.next;
    return lessHead.next;
  }

  // 面试题 02.05. 链表求和
  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    int ans = 0, index = 1, tmp, carry = 0;
    ListNode dummyHead = new ListNode(-1);
    ListNode c = dummyHead;
    while (l1 != null && l2 != null) {
      tmp = l1.val + l2.val + carry;
      carry = tmp / 10;
      c.next = new ListNode(tmp % 10);
      c = c.next;
      l1 = l1.next;
      l2 = l2.next;
    }
    while (l1 != null) {
      tmp = l1.val + carry;
      carry = tmp / 10;
      c.next = new ListNode(tmp % 10);
      c = c.next;
      l1 = l1.next;
    }
    while (l2 != null) {
      tmp = l2.val + carry;
      carry = tmp / 10;
      c.next = new ListNode(tmp % 10);
      c = c.next;
      l2 = l2.next;
    }
    if (carry == 1) {
      c.next = new ListNode(1);
    }
    return dummyHead.next;
  }

  // 面试题 02.06. 回文链表
  public boolean isPalindrome(ListNode head) {
    if (head == null) {
      return true;
    }
    var slow = head;
    // fast 如果初始化为 head.Next, 则中点在 slow.Next
    // fast 初始化为 head, 则中点在 slow
    var fast = head.next;
    while (fast != null && fast.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }

    var tail = reverseList(slow.next);
    // 断开两个链表(需要用到中点前一个节点)
    slow.next = null;
    while (head != null && tail != null) {
      if (head.val != tail.val) {
        return false;
      }
      head = head.next;
      tail = tail.next;
    }
    return true;
  }

  private ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }
    ListNode last = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return last;
  }

  // 面试题 02.07. 链表相交
  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode la = headA;
    ListNode lb = headB;
    while (la != lb) {
      la = la == null ? headB : la.next;
      lb = lb == null ? headA : lb.next;
    }
    return la;

  }

  // 面试题 02.08. 环路检测
  public ListNode detectCycle(ListNode head) {
    if (head == null || head.next == null) {
      return null;
    }
    var slow = head;
    var fast = head.next;
    while (fast != null && fast.next != null) {
      if (fast == slow) { // 有环
        fast = head;
        slow = slow.next;
        while (fast != slow) {
          fast = fast.next;
          slow = slow.next;
        }
        return slow;
      }
      slow = slow.next;
      fast = fast.next.next;
    }
    return null;
  }

  // 面试题 04.01. 节点间通路
  public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
    // TODO Graph
    return false;
  }

  // 面试题 04.02. 最小高度树
  public TreeNode sortedArrayToBST(int[] nums) {
    int n = nums.length;
    return sortedArrayToBSTHelper(nums, 0, n - 1);
  }

  private TreeNode sortedArrayToBSTHelper(int[] nums, int start, int end) {
    if (start > end) {
      return null;
    }
    if (start == end) {
      return new TreeNode(nums[start]);
    }
    int mid = start + (end - start) / 2;
    TreeNode root = new TreeNode(nums[mid]);
    root.left = sortedArrayToBSTHelper(nums, start, mid - 1);
    root.right = sortedArrayToBSTHelper(nums, mid + 1, end);
    return root;
  }

  // 面试题 04.03. 特定深度节点链表
  public ListNode[] listOfDepth(TreeNode tree) {
    ArrayList<ListNode> ans = new ArrayList<>();
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(tree);
    while (!queue.isEmpty()) {
      int size = queue.size();
      ListNode dummyHead = new ListNode(-1);
      ListNode curr = dummyHead;
      for (int i = 0; i < size; i += 1) {
        var curNode = queue.poll();
        assert curNode != null;
        curr.next = new ListNode(curNode.val);
        curr = curr.next;
        if (curNode.left != null) {
          queue.offer(curNode.left);
        }
        if (curNode.right != null) {
          queue.offer(curNode.right);
        }
      }
      ans.add(dummyHead.next);
    }
    return ans.toArray(new ListNode[0]);

  }

  // 面试题 04.04. 检查平衡性
  public boolean isBalanced(TreeNode root) {
    return height(root) >= 0;
  }

  private int height(TreeNode root) {
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

  // 面试题 04.05. 合法二叉搜索树
  // 中序遍历
  // 递归，需要考虑“上下界”
  public boolean isValidBST(TreeNode root) {
    Deque<TreeNode> stack = new LinkedList<>();
    double prev = -Double.MAX_VALUE;

    while (!stack.isEmpty() || root != null) {
      while (root != null) {
        stack.push(root);
        root = root.left;
      }
      root = stack.pop();
      if (root.val <= prev) {
        return false;
      }
      prev = root.val;
      root = root.right;
    }
    return true;
  }

  // 面试题 04.06. 后继者
  public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
    TreeNode pre = null;
    while (root.val != p.val) {
      if (p.val > root.val) {
        root = root.right;
      } else {
        pre = root;
        root = root.left;
      }
    }
    if (root.right == null) {
      return pre;
    }
    root = root.right;
    while (root.left != null) {
      root = root.left;
    }
    return root;
  }

  // 面试题 04.08. 首个共同祖先
  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null) {
      return null;
    }
    if (root == p || root == q) {
      return root;
    }
    var left = lowestCommonAncestor(root.left, p, q);
    var right = lowestCommonAncestor(root.right, p, q);

    if (left != null && right != null) {
      return root;
    }
    if (left != null) {
      return left;
    }
    return right;
  }

  // TODO 面试题 04.09. 二叉搜索树序列

  // 面试题 10.01. 合并排序的数组
  public void merge(int[] A, int m, int[] B, int n) {
    int pa = m - 1, pb = n - 1;
    int tail = m + n - 1;
    int cur;
    while (pa >= 0 || pb >= 0) {
      if (pa == -1) {
        cur = B[pb--];
      } else if (pb == -1) {
        cur = A[pa--];
      } else if (A[pa] > B[pb]) {
        cur = A[pa--];
      } else {
        cur = B[pb--];
      }
      A[tail--] = cur;
    }
  }

  // 面试题 10.03. 搜索旋转数组
  public int search(int[] arr, int target) {
    // TODO
    return -1;
  }

  // 面试题 10.05. 稀疏数组搜索
  public int findString(String[] words, String s) {
    int n = words.length;
    if (n == 0) {
      return -1;
    }
    int left = 0, right = n - 1;
    while (left < right) {
      int m = left + (right - left) / 2;
      int temp = m;
      // 排除非空字符
      while (m <= right && words[m].equals("")) {
        m++;
      }
      if (m > right) {
        right = temp - 1;
        continue;
      }
      if (words[m].equals(s)) {
        return m;
      } else if (words[m].compareTo(s) < 0) {
        left = m + 1;
      } else if (words[m].compareTo(s) > 0) {
        right = m - 1;
      }
    }
    if (left < n && words[left].equals(s)) {
      return left;
    }
    return -1;
  }

  // 面试题 10.09. 排序矩阵查找
  public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
      return false;
    }
    int rows = matrix.length, cols = matrix[0].length;
    int i = 0, j = cols - 1;
    while (i < rows && j > -1) {
      if (matrix[i][j] == target) {
        return true;
      } else if (matrix[i][j] < target) {
        i++;
      } else {
        j--;
      }
    }
    return false;
  }

  // 面试题 16.01 - 交换数字
  public int[] swapNumbers(int[] numbers) {
    numbers[0] = numbers[0] ^ numbers[1];
    numbers[1] = numbers[0] ^ numbers[1];
    numbers[0] = numbers[0] ^ numbers[1];
    return numbers;
  }

  // 面试题 16.05 - 阶乘尾数
  public int trailingZeroes(int n) {
    int res = 0;
    while (n >= 5) {
      res += n / 5;
      n /= 5;
    }
    return res;
  }

  // 面试题 16.06 - 最小差
  public int smallestDifference(int[] a, int[] b) {
    Arrays.sort(a);
    Arrays.sort(b);
    int i = 0, j = 0;
    long ans = Long.MAX_VALUE;
    while (i < a.length && j < b.length) {
      if (a[i] == b[j]) {
        return 0;
      } else if (a[i] > b[j]) {
        ans = Math.min((long) a[i] - (long) b[j], ans);
        j++;
      } else {
        ans = Math.min((long) b[j] - (long) a[i], ans);
        i++;
      }
    }
    return (int) ans;
  }

  // 面试题 16.07 - 最大数值
  public int maximum(int a, int b) {
    long diff = (long) a - (long) b;
    return (int) (((long) a + (long) b + (diff ^ (diff >> 63)) - (diff >> 63)) >> 1);
  }

}
