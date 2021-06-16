package leetcode;

import java.util.*;

class TreeSolution {

    // 872 easy 叶子相似的树
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> seq1 = new ArrayList<>();
        if (root1 != null) {
            dfsLeftSimilar(root1, seq1);
        }

        List<Integer> seq2 = new ArrayList<>();
        if (root2 != null) {
            dfsLeftSimilar(root2, seq2);
        }

        return seq1.equals(seq2);
    }

    public void dfsLeftSimilar(TreeNode node, List<Integer> seq) {
        if (node.left == null && node.right == null) {
            seq.add(node.val);
        } else {
            if (node.left != null) {
                dfsLeftSimilar(node.left, seq);
            }
            if (node.right != null) {
                dfsLeftSimilar(node.right, seq);
            }
        }
    }

    // 450 medium 删除二叉搜索树中的节点
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }

        // delete from the right subtree
        if (key > root.val) {
            root.right = deleteNode(root.right, key);
        }
        // delete from the left subtree
        else if (key < root.val) {
            root.left = deleteNode(root.left, key);
        }
        // delete the current node
        else {
            // the node is a leaf
            if (root.left == null && root.right == null) {
                root = null;
            }
            // the node is not a leaf and has a right child
            else if (root.right != null) {
                root.val = successor(root);
                root.right = deleteNode(root.right, root.val);
            }
            // the node is not a leaf, has no right child, and has a left child
            else {
                root.val = predecessor(root);
                root.left = deleteNode(root.left, root.val);
            }
        }
        return root;
    }

    /*
    One step right and then always left
    */
    public int successor(TreeNode root) {
        root = root.right;
        while (root.left != null) {
            root = root.left;
        }
        return root.val;
    }

    /*
    One step left and then always right
    */
    public int predecessor(TreeNode root) {
        root = root.left;
        while (root.right != null) {
            root = root.right;
        }
        return root.val;
    }

    // 701
    // 输入数据保证，新值和原始二叉搜索树中的任意节点值都不同。
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            root = new TreeNode(val);
            return root;
        }
        if (root.val > val) {
            root.left = insertIntoBST(root.left, val);
        } else {
            root.right = insertIntoBST(root.right, val);
        }
        return root;
    }

    // 104
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        var left = maxDepth(root.left);
        var right = maxDepth(root.right);

        return left > right ? left : right + 1;
    }

    // 236
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

    // 938 二叉搜索树的范围和 easy
    public int rangeSumBST(TreeNode root, int low, int high) {
        if (root == null) {
            return 0;
        }
        if (root.val > high) {
            return rangeSumBST(root.left, low, high);
        }
        if (root.val < low) {
            return rangeSumBST(root.right, low, high);
        }
        return root.val + rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high);
    }

    // 109
    public TreeNode sortedListToBST(ListNode head) {
        return buildTree(head, null);
    }

    public TreeNode buildTree(ListNode left, ListNode right) {
        if (left == right) {
            return null;
        }
        ListNode mid = getMedian(left, right);
        TreeNode root = new TreeNode(mid.val);
        root.left = buildTree(left, mid);
        root.right = buildTree(mid.next, right);
        return root;
    }

    public ListNode getMedian(ListNode left, ListNode right) {
        // 快慢指针
        ListNode fast = left;
        ListNode slow = left;
        // fast 与 right 比较
        while (fast != right && fast.next != right) {
            fast = fast.next;
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // 103
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        nodeQueue.offer(root);
        int curCount = 1, nextCount = 0;
        boolean toRight = true;
        while (!nodeQueue.isEmpty() && curCount > 0) {
            var levelRes = new LinkedList<Integer>();
            while (curCount > 0) {
                var curNode = nodeQueue.poll();
                assert curNode != null;
                if (toRight) {
                    levelRes.offerLast(curNode.val);
                } else {
                    levelRes.offerFirst(curNode.val);
                }
                if (curNode.left != null) {
                    nodeQueue.add(curNode.left);
                    nextCount++;
                }
                if (curNode.right != null) {
                    nodeQueue.add(curNode.right);
                    nextCount++;
                }
                curCount--;
            }
            res.add(levelRes);
            curCount = nextCount;
            nextCount = 0;
            toRight = !toRight;
        }

        return res;
    }

    // 114 二叉树展开为链表
    public void flattenNonRecurse(TreeNode root) {
        // 先序遍历
        List<TreeNode> list = new ArrayList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode node = root;
        while (node != null || !stack.isEmpty()) {
            while (node != null) {
                list.add(node);
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            node = node.right;
        }
        int size = list.size();
        for (int i = 1; i < size; i++) {
            TreeNode prev = list.get(i - 1), curr = list.get(i);
            prev.left = null;
            prev.right = curr;
        }
    }

    // 95
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<>();
        }
        return generateTrees(1, n);
    }

    List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> allTrees = new LinkedList<>();
        if (start > end) {
            allTrees.add(null);
            return allTrees;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> leftTrees = generateTrees(start, i - 1);
            List<TreeNode> rightTrees = generateTrees(i + 1, end);

            for (TreeNode left : leftTrees) {
                for (TreeNode right : rightTrees) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    allTrees.add(root);
                }
            }
        }
        return allTrees;
    }


    // 96
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    // 654
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return build(nums, 0, nums.length - 1);
    }

    TreeNode build(int[] nums, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        int maxIndex = -1, maxVal = Integer.MIN_VALUE;
        for (int i = lo; i <= hi; i++) {
            if (maxVal < nums[i]) {
                maxIndex = i;
                maxVal = nums[i];
            }
        }
        TreeNode root = new TreeNode(maxVal);
        root.left = build(nums, lo, maxIndex - 1);
        root.right = build(nums, maxIndex + 1, hi);
        return root;
    }

    // 783
    public int minDiffInBST(TreeNode root) {
        int min = Integer.MAX_VALUE;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root, prev = null;
        while (cur != null || !stack.empty()) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop();
                if (prev != null) {
                    min = Math.min(min, cur.val - prev.val);
                }
                prev = cur;
                cur = cur.right;
            }
        }
        return min;
    }

    // 99
    public void recoverTree(TreeNode root) {
        List<TreeNode> list = new ArrayList<>();
        dfs(root, list);
        TreeNode x = null;
        TreeNode y = null;
        for (int i = 0; i < list.size() - 1; ++i) {
            if (list.get(i).val > list.get(i + 1).val) {
                y = list.get(i + 1);
                if (x == null) {
                    x = list.get(i);
                }
            }
        }
        if (x != null && y != null) {
            int tmp = x.val;
            x.val = y.val;
            y.val = tmp;
        }
    }

    private void dfs(TreeNode node, List<TreeNode> list) {
        if (node == null) {
            return;
        }
        dfs(node.left, list);
        list.add(node);
        dfs(node.right, list);
    }

    // 105 从前序与中序遍历序列构造二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || preorder.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[0]);
        Deque<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        int inorderIndex = 0;
        for (int i = 1; i < preorder.length; i++) {
            int preorderVal = preorder[i];
            TreeNode node = stack.peek();
            assert node != null;
            if (node.val != inorder[inorderIndex]) {
                // 左子树
                node.left = new TreeNode(preorderVal);
                stack.push(node.left);
            } else {
                // 右子树
                while (!stack.isEmpty() && stack.peek().val == inorder[inorderIndex]) {
                    node = stack.pop();
                    inorderIndex++;
                }
                node.right = new TreeNode(preorderVal);
                stack.push(node.right);
            }
        }
        return root;
    }

    // 222
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftLvl = countLevel(root.left);
        int rightLvl = countLevel(root.right);
        if (leftLvl == rightLvl) {
            return countNodes(root.right) + (1 << leftLvl);
        } else {
            return countNodes(root.left) + (1 << rightLvl);
        }
    }

    private int countLevel(TreeNode root) {
        int level = 0;
        while (root != null) {
            level++;
            root = root.left;
        }
        return level;
    }

    // 111
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left == null && root.right == null) {
            return 1;
        }

        var minimumDepth = Integer.MAX_VALUE;
        if (root.left != null) {
            minimumDepth = Math.min(minDepth(root.left), minimumDepth);
        }
        if (root.right != null) {
            minimumDepth = Math.min(minDepth(root.right), minimumDepth);
        }

        return minimumDepth + 1;
    }

    // 404
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left != null && root.left.left == null && root.left.right == null) {
            return root.left.val + sumOfLeftLeaves(root.right);
        }
        return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
    }

    // 108
    public TreeNode sortedArrayToBST(int[] nums) {
        return arrayToBSTHelper(nums, 0, nums.length - 1);
    }

    TreeNode arrayToBSTHelper(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }

        int mid = (left + right) / 2;

        var ret = new TreeNode(nums[mid]);
        ret.left = arrayToBSTHelper(nums, left, mid - 1);
        ret.right = arrayToBSTHelper(nums, mid + 1, right);
        return ret;
    }

    // 572
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if (t == null) {
            return true;
        }
        if (s == null) {
            return false;
        }
        return isSubtree(s.left, t) || isSubtree(s.right, t) || isSameTree(s, t);
    }

    public boolean isSameTree(TreeNode s, TreeNode t) {
        if (s == null && t == null) {
            return true;
        }
        if (s == null || t == null) {
            return false;
        }
        if (s.val != t.val) {
            return false;
        }
        return isSameTree(s.left, t.left) && isSameTree(s.right, t.right);
    }

    // 897
    public TreeNode increasingBST(TreeNode root) {
        var values = new ArrayList<Integer>();
        inorderTraversal(root, values);
        TreeNode ans = new TreeNode(0), cur = ans;
        for (int v : values) {
            cur.right = new TreeNode(v);
            cur = cur.right;
        }
        return ans.right;
    }

    public void inorderTraversal(TreeNode node, List<Integer> values) {
        if (node == null) {
            return;
        }
        inorderTraversal(node.left, values);
        values.add(node.val);
        inorderTraversal(node.right, values);
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return root.val == targetSum;
        }
        return hasPathSum(root.left, targetSum - root.val) ||
                hasPathSum(root.right, targetSum - root.val);
    }

    // 101
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
        return (t1.val == t2.val)
                && isMirror(t1.left, t2.right)
                && isMirror(t1.right, t2.left);
    }

    // 98
    public boolean isValidBST(TreeNode root) {
        if (root == null || root.left == null && root.right == null) {
            return true;
        }
        var p = root;
        // 中序遍历
        Stack<TreeNode> stack = new Stack<>();
        var nums = new LinkedList<Integer>();
        while (!stack.empty() || p != null) {
            while (p != null) {
                stack.add(p);
                p = p.left;
            }
            var tmp = stack.pop();
            nums.addLast(tmp.val);
            p = tmp.right;
        }
        var arr = nums.toArray();
        for (int i = 0; i < arr.length - 1; i++) {
            if ((Integer) arr[i] >= (Integer) arr[i + 1]) {
                return false;
            }
        }
        return true;
    }

    // 107
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int curCount = 1, nextCount = 0;
        while (!queue.isEmpty() && curCount > 0) {
            var temp = new LinkedList<Integer>();
            while (curCount > 0) {
                var head = queue.peek();
                assert head != null;
                if (head.left != null) {
                    queue.add(head.left);
                    nextCount++;
                }
                if (head.right != null) {
                    queue.add(head.right);
                    nextCount++;
                }
                temp.addLast(head.val);
                curCount--;
                queue.poll();
            }
            res.add(0, temp);
            curCount = nextCount;
            nextCount = 0;
        }

        return res;
    }

    // 110
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
}
