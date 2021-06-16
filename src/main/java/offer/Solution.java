package offer;

import java.util.*;
import java.util.stream.Collectors;

class Solution {

    // 03. 数组中重复的数字
    // 使用 HashSet
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

    // 04. 二维数组的查找
    // 在一个二维数组中，
    // 每一行都按照从左到右递增的顺序排序，
    // 每一列都按照从上到下递增的顺序排序。
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

    // 05. 替换空格
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

    // 06. 从尾到头打印列表
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

    // 08. 二叉树的下一个节点（节点中包含指向父节点的指针）
    public TreeLinkNode getNext(TreeLinkNode node) {
        if (node == null) {
            return null;
        }
        if (node.right != null) {
            var rNode = node.right;
            while (rNode.left != null) {
                rNode = rNode.left;
            }
            return rNode;
        } else {
            while (node.parent != null) {
                var pNode = node.parent;
                if (pNode.left == node) {
                    return pNode;
                }
                node = node.parent;
            }
        }
        return null;
    }

    // 10-I. 斐波那契数列
    public int fib(int n) {
        if (n == 0 || n == 1) {
            return n;
        }
        int a = 0, b = 1, tmp;
        while (n > 1) {
            tmp = a;
            a = b;
            b = (a + tmp) % 1000000007;
            n--;
        }
        return b;
    }

    // 10-II. 青蛙跳台阶
    public int numWays(int n) {
        if (n < 2) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007;
        }

        return dp[n];
    }

    // 11. 旋转数组的最小数字
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

    // 12. 矩阵中的路径
    public boolean exist(char[][] board, String word) {
        char[] words = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, words, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, char[] word, int i, int j, int matched) {
        if (i >= board.length || i < 0 || j >= board[0].length || j < 0
                || board[i][j] != word[matched]) {
            return false;
        }
        if (matched == word.length - 1) {
            return true;
        }
        board[i][j] = '\0';
        boolean res =
                dfs(board, word, i + 1, j, matched + 1) || dfs(board, word, i - 1, j, matched + 1) ||
                        dfs(board, word, i, j + 1, matched + 1) || dfs(board, word, i, j - 1, matched + 1);
        board[i][j] = word[matched]; // 还原
        return res;
    }

    // 13. 机器人的运动范围 BFS
    public int movingCount(int m, int n, int k) {
        if (k == 0) {
            return 1;
        }
        Queue<int[]> queue = new LinkedList<>();
        // 向右和向下的方向数组
        int[] dx = {0, 1};
        int[] dy = {1, 0};
        boolean[][] visited = new boolean[m][n];
        queue.offer(new int[]{0, 0});
        visited[0][0] = true;
        int ans = 1;
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int x = cell[0], y = cell[1];
            for (int i = 0; i < 2; ++i) {
                int tx = dx[i] + x;
                int ty = dy[i] + y;
                if (tx < 0 || tx >= m || ty < 0 || ty >= n || visited[tx][ty]
                        || sumBits(tx) + sumBits(ty) > k) {
                    continue;
                }
                queue.offer(new int[]{tx, ty});
                visited[tx][ty] = true;
                ans++;
            }
        }
        return ans;
    }

    private int sumBits(int x) {
        int res = 0;
        while (x != 0) {
            res += x % 10;
            x /= 10;
        }
        return res;
    }

    // 14-I. 剪绳子
    // 2 <= n <= 58
    public int cuttingRopeDP(int n) {
        if (n == 2) {
            return 1;
        }
        if (n == 3) {
            return 2;
        }
        int timesOfThree = n / 3;

        if (n % 3 == 1) {
            timesOfThree--;
        }
        int timesOfTwo = (n - timesOfThree * 3) / 2;

        return (int) (Math.pow(3, timesOfThree) * Math.pow(2, timesOfTwo));
    }

    // 15. 二进制中 1 的个数
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n = n & (n - 1);
            count++;
        }
        return count;
    }

    // 16. 数值的整数次方
    // 快速幂
    public double myPow(double x, int n) {
        if (x == 0) {
            return 0;
        }
        long b = n;
        double res = 1.0;
        if (b < 0) {
            x = 1 / x;
            b = -b;
        }
        while (b > 0) {
            if ((b & 1) == 1) {
                res *= x;
            }
            x *= x;
            b >>= 1;
        }
        return res;
    }

    // 17. 打印从 1 到最大的 n 位数
    public int[] printNumbers(int n) {
        int num = (int) Math.pow(10, n);
        int[] result = new int[num - 1];
        for (int i = 0; i < num - 1; i++) {
            result[i] = i + 1;
        }
        return result;
    }

    // 18. 删除链表的节点
    public ListNode deleteNode(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        if (head.val == val) {
            return head.next;
        }
        head.next = deleteNode(head.next, val);
        return head;
    }

    // 21. 调整数组顺序使奇数位于偶数前面
    public int[] exchange(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        while (l < r) {
            if ((nums[l] & 1) == 1 && (nums[r] & 1) == 0) {
                l++;
                r--;
            } else if ((nums[l] & 1) == 0 && (nums[r] & 1) == 1) {
                var tmp = nums[l];
                nums[l] = nums[r];
                nums[r] = tmp;
                l++;
                r--;
            } else if ((nums[l] & 1) == 1) {
                l++;
            } else {
                r--;
            }
        }
        return nums;
    }

    // 22. 链表中倒数第k个节点
    public ListNode getKthFromEnd(ListNode head, int k) {
        var fast = head;
        var slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // 24. 反转链表
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        var ret = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return ret;
    }

    // 25. 合并两个排序的链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        var dummyHead = new ListNode();
        var c = dummyHead;
        var c1 = l1;
        var c2 = l2;
        while (c1 != null && c2 != null) {
            if (c1.val <= c2.val) {
                c.next = c1;
                c = c.next;
                c1 = c1.next;
            } else {
                c.next = c2;
                c = c.next;
                c2 = c2.next;
            }
        }
        if (c1 != null) {
            c.next = c1;
        }
        if (c2 != null) {
            c.next = c2;
        }
        return dummyHead.next;
    }

    // 26. 树的子结构
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return (A != null && B != null) && (recur(A, B) || isSubStructure(A.left, B) ||
                isSubStructure(A.right, B));
    }

    private boolean recur(TreeNode A, TreeNode B) {
        if (B == null) {
            return true;
        }
        if (A == null || A.val != B.val) {
            return false;
        }
        return recur(A.left, B.left) && recur(A.right, B.right);
    }

    // 27. 二叉树的镜像
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

    // 28. 对称的二叉树
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    private boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null || t1.val != t2.val) {
            return false;
        }
        return isMirror(t1.left, t2.right) && isMirror(t1.right, t2.left);
    }

    // 29. 顺时针打印矩阵
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

    // 31. 栈的压入、弹出序列
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stack = new LinkedList<>();
        int index = 0;
        for (var i : pushed) {
            stack.push(i);
            while (!stack.isEmpty() && stack.peek() == popped[index]) {
                stack.pop();
                index++;
            }
        }
        return index == popped.length;
    }

    // 32-I. 从上到下打印二叉树
    public int[] levelOrder(TreeNode root) {
        if (root == null) {
            return new int[0];
        }
        Queue<TreeNode> queue = new LinkedList<>() {{
            add(root);
        }};
        ArrayList<Integer> ans = new ArrayList<>();
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            ans.add(node.val);
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
        int[] res = new int[ans.size()];
        for (int i = 0; i < ans.size(); i++) {
            res[i] = ans.get(i);
        }
        return res;
    }

    // 32-II. 从上到下打印二叉树 II
    public List<List<Integer>> levelOrderTwo(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root != null) {
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            int count = queue.size();
            for (int i = count; i > 0; i--) {
                TreeNode node = queue.poll();
                assert node != null;
                tmp.add(node.val);
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            res.add(tmp);
        }
        return res;
    }

    // 32-III. 从上到下打印二叉树 III
    public List<List<Integer>> levelOrderThree(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root != null) {
            deque.add(root);
        }
        while (!deque.isEmpty()) {
            // 打印奇数层
            List<Integer> tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                // 从左向右打印
                TreeNode node = deque.removeFirst();
                tmp.add(node.val);
                if (node.left != null) {
                    deque.addLast(node.left);
                }
                if (node.right != null) {
                    deque.addLast(node.right);
                }
            }
            res.add(tmp);
            if (deque.isEmpty()) {
                break; // 提前跳出
            }
            // 打印偶数层
            tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                // 从右向左打印
                TreeNode node = deque.removeLast();
                tmp.add(node.val);
                if (node.right != null) {
                    deque.addFirst(node.right);
                }
                if (node.left != null) {
                    deque.addFirst(node.left);
                }
            }
            res.add(tmp);
        }
        return res;
    }

    // 33. 二叉搜索树的后序遍历序列
    public boolean verifyPostorder(int[] postorder) {
        if (postorder == null || postorder.length == 0) {
            return true;
        }

        return verify(postorder, 0, postorder.length - 1);
    }

    private boolean verify(int[] sequence, int first, int last) {
        if (first - last == 0) {
            return true;
        }
        int right = first;
        while (sequence[right] < sequence[last] && right < last) {
            right++;
        }
        int left = right;
        while (left < last) {
            if (sequence[left] < sequence[last]) {
                return false;
            }
            left++;
        }
        // 保证不越界
        boolean leftTree = true;
        if (right - 1 > first) {
            leftTree = verify(sequence, first, right - 1);
        }
        boolean rightTree = true;
        if (last - 1 > right) {
            rightTree = verify(sequence, right, last - 1);
        }
        return leftTree && rightTree;
    }

    // 39. 数组中出现次数超过一半的数字
    public int majorityElement(int[] nums) {
        // 排序
        Arrays.sort(nums);
        return nums[nums.length >> 1];
    }

    // 摩尔投票法
    public int majorityElementV2(int[] nums) {
        int candidate = nums[0], count = 1;
        for (int i = 1; i < nums.length; ++i) {
            if (candidate == nums[i]) {
                ++count;
            } else if (--count == 0) {
                candidate = nums[i];
                count = 1;
            }
        }
        return candidate;
    }

    // 40. 最小的 k 个数
    public int[] getLeastNumbers(int[] arr, int k) {
        int[] res = new int[k];
        if (k == 0) { // 排除 0 的情况
            return res;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>((num1, num2) -> num2 - num1);
        for (int i = 0; i < k; ++i) {
            queue.offer(arr[i]);
        }
        for (int i = k; i < arr.length; ++i) {
            if (queue.peek() > arr[i]) {
                queue.poll();
                queue.offer(arr[i]);
            }
        }
        for (int i = 0; i < k; ++i) {
            res[i] = queue.poll();
        }
        return res;
    }

    // 42. 连续子数组的最大和
    public int maxSubArray(int[] nums) {
        int ans = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nums[i] += Math.max(nums[i - 1], 0);
            ans = Math.max(ans, nums[i]);
        }
        return ans;
    }

    // 44. 第 N 位数字
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while (n > count) {
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit;
        return Long.toString(num).charAt((n - 1) % digit) - '0';
    }

    // 45. 把数组排成最小的数
    public String minNumber(int[] nums) {
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strings, (x, y) -> (x + y).compareTo(y + x));
        StringBuilder res = new StringBuilder();
        for (String s : strings) {
            res.append(s);
        }
        return res.toString();
    }

    // 46. 把数字翻译成字符串
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int a = 1, b = 1;
        for (int i = 2; i <= s.length(); i++) {
            String tmp = s.substring(i - 2, i);
            int c = tmp.compareTo("10") >= 0 && tmp.compareTo("25") <= 0 ? a + b : a;
            b = a;
            a = c;
        }
        return a;
    }

    // 47. 礼物的最大价值
    public int maxValue(int[][] grid) {
        int rows = grid.length, cols = grid[0].length;
        int[][] dp = new int[rows][cols];
        dp[0][0] = grid[0][0];
        for (int j = 1; j < cols; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < rows; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[rows - 1][cols - 1];
    }

    // 48. 最长不含重复字符的子字符串
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

    // 49. 丑数
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

    // 50. 第一个只出现一次的字符
    public char firstUniqChar(String s) {
        HashMap<Character, Boolean> dic = new HashMap<>();
        char[] sc = s.toCharArray();
        for (char c : sc) {
            dic.put(c, !dic.containsKey(c));
        }
        for (char c : sc) {
            if (dic.get(c)) {
                return c;
            }
        }
        return ' ';
    }

    // 52. 两个链表的第一个公共节点
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        var t1 = headA;
        var t2 = headB;
        while (t1 != t2) {
            t1 = t1 == null ? headB : t1.next;
            t2 = t2 == null ? headA : t2.next;
        }
        return t1;
    }

    // 53-I. 在排序数组中查找数字 I
    public int search(int[] nums, int target) {
        int n = nums.length, l = 0, r = n, mid = n / 2;
        while (l < r) {
            mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                break;
            } else if (target < nums[mid]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        if (l >= r) {
            return 0;
        }
        int count = 1;
        l = mid - 1;
        while (l > -1 && nums[l] == target) {
            count++;
            l--;
        }
        r = mid + 1;
        while (r < n && nums[r] == target) {
            count++;
            r++;
        }
        return count;
    }

    // 53-II. 0~n 中缺失的数字
    public int missingNumber(int[] nums) {
        int l = 0, r = nums.length - 1, m;
        while (l <= r) {
            m = l + (r - l) / 2;
            if (nums[m] == m) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return l;
    }

    // 55-I. 二叉树的深度
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

    // 55-II. 平衡二叉树
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

    // 56-I. 数组中数字出现的次数
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

    // 57. 和为 s 的两个数字
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

    // 57-II. 何为 s 的连续正数序列
    public int[][] findContinuousSequence(int target) {
        int i = 1; // 滑动窗口的左边界
        int j = 1; // 滑动窗口的右边界
        int sum = 0; // 滑动窗口中数字的和
        List<int[]> res = new ArrayList<>();

        while (i <= target / 2) {
            if (sum < target) {
                sum += j; // 右边界向右移动
                j++;
            } else if (sum > target) {
                sum -= i; // 左边界向右移动
                i++;
            } else {
                int[] arr = new int[j - i];
                for (int k = i; k < j; k++) {
                    arr[k - i] = k;
                }
                res.add(arr);
                sum -= i; // 左边界向右移动
                i++;
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    // 58. 翻转字符串里的单词
    public String reverseWords(String s) {
        var tmp = Arrays.stream(s.strip().split(" ")).filter(w -> w.length() > 0).collect(
                Collectors.toList());
        Collections.reverse(tmp);
        return String.join(" ", tmp);
    }

    // 58-II. 左旋转字符串
    public String reverseLeftWords(String s, int n) {
        StringBuilder res = new StringBuilder();
        for (int i = n; i < s.length(); i++) {
            res.append(s.charAt(i));
        }
        for (int i = 0; i < n; i++) {
            res.append(s.charAt(i));
        }
        return res.toString();
    }

    // 60. n 个色子的点数
    // 概率论
    public double[] dicesProbability(int n) {
        double[] ans = new double[5 * n + 1];
        double[][] dp = new double[n + 1][6 * n + 1];
        for (int i = 1; i <= 6; i++) {
            dp[1][i] = 1.0 / 6.0;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = i; j <= 6 * n; j++) {
                for (int k = 1; k <= 6; k++) {
                    if (j > k) {
                        dp[i][j] += dp[i - 1][j - k] * dp[1][k];
                    }
                }
            }
        }
        System.arraycopy(dp[n], n, ans, 0, 6 * n + 1 - n);
        return ans;
    }

    // 61. 扑克牌中的顺子
    public boolean isStraight(int[] nums) {
        Set<Integer> s = new HashSet<>();
        int max = 0, min = 14;
        for (var num : nums) {
            if (num == 0) {
                continue;
            }
            if (s.contains(num)) {
                return false;
            }
            if (num > max) {
                max = num;
            }
            if (num < min) {
                min = num;
            }
            s.add(num);
        }
        return max - min < 5;
    }

    // 62. 圆圈中最后剩下的数字
    public int lastRemaining(int n, int m) {
        int ans = 0;
        // f(n, m) = (f(n-1, m) + m) % n
        for (int i = 2; i <= n; i++) {
            ans = (ans + m) % i;
        }
        return ans;
    }

    // 63. 股票的最大利润 medium
    public int maxProfit(int[] prices) {
        int cost = Integer.MAX_VALUE, profit = 0;
        for (int price : prices) {
            cost = Math.min(cost, price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }

    // 64. 求 1 + 2 + ... + n
    public int sumNums(int n) {
        // 利用 && 短路
        boolean x = n > 1 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    // 65. 不用 加减乘除 做 加法
    public int add(int a, int b) {
        while (b != 0) { // 当进位为 0 时跳出
            int c = (a & b) << 1;  // c = 进位
            a ^= b; // a = 非进位和
            b = c; // b = 进位
        }
        return a;
    }

    // 66. 构建乘积数组
    public int[] constructArr(int[] a) {
        if (a.length == 0) {
            return new int[0];
        }
        int[] b = new int[a.length];
        b[0] = 1;
        int tmp = 1;
        for (int i = 1; i < a.length; i++) {
            b[i] = b[i - 1] * a[i - 1];
        }
        for (int i = a.length - 2; i >= 0; i--) {
            tmp *= a[i + 1];
            b[i] *= tmp;
        }
        return b;
    }

    // 67. 把字符串转换成整数
    public int strToInt(String str) {
        int res = 0, boundary = Integer.MAX_VALUE / 10;
        int i = 0, sign = 1, length = str.length();
        if (length == 0) {
            return 0;
        }
        while (str.charAt(i) == ' ') {
            if (++i == length) {
                return 0;
            }
        }
        if (str.charAt(i) == '-') {
            sign = -1;
        }
        if (str.charAt(i) == '-' || str.charAt(i) == '+') {
            i++;
        }
        for (int j = i; j < length; j++) {
            if (str.charAt(j) < '0' || str.charAt(j) > '9') {
                break;
            }
            if (res > boundary || res == boundary && str.charAt(j) > '7') {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            res = res * 10 + (str.charAt(j) - '0');
        }
        return sign * res;
    }

    // 68-I. 二叉搜索树的最近公共祖先
    public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestorBST(root.left, p, q);
        }
        if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestorBST(root.right, p, q);
        }
        return root;
    }

    // 68-II. 二叉树的最近公共祖先
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
}
