package algo;

import java.util.*;

class TreeTraverse {

    /**
     * 先序遍历 递归
     *
     * @param root 根节点
     */
    public void preorderTraversal(TreeNode root) {
        if (root == null) {
            return;
        }
        System.out.println(root.val);
        preorderTraversal(root.left);
        preorderTraversal(root.right);
    }

    /**
     * 先序遍历 非递归
     *
     * @param root 根节点
     * @return 遍历结果
     */
    public List<Integer> preorderTraversalNonRec(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                // 先序遍历，先保存结果
                result.add(node.val);
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            node = node.right;
        }
        return result;
    }

    /**
     * 中序遍历 递归
     *
     * @param root 根节点
     */
    public void inorderTraversal(TreeNode root) {
        if (root == null) {
            return;
        }
        preorderTraversal(root.left);
        System.out.println(root.val);
        preorderTraversal(root.right);
    }

    /**
     * 中序遍历 非递归
     *
     * @param root 根节点
     * @return 遍历结果
     */
    public List<Integer> inorderTraversalNonRec(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left; // 一直向左
            }
            root = stack.pop();
            result.add(root.val); // 保存结果
            root = root.right;
        }
        return result;
    }

    /**
     * 后序遍历 递归
     *
     * @param root 根节点
     */
    public void postorderTraversal(TreeNode root) {
        if (root == null) {
            return;
        }
        preorderTraversal(root.left);
        preorderTraversal(root.right);
        System.out.println(root.val);
    }

    /**
     * 后序遍历 非递归
     *
     * @param root 根节点
     * @return 遍历结果
     */
    public List<Integer> postorderTraversalNonRec(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }

        Deque<TreeNode> stack = new LinkedList<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                result.add(root.val);
                stack.push(root);
                root = root.right;
            }

            TreeNode cur = stack.pop();
            root = cur.left;
        }

        Collections.reverse(result);
        return result;
    }

    /**
     * 层序遍历
     *
     * @param root 根节点
     * @return 遍历结果
     */
    public List<Integer> levelOrderTraverse(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            res.add(node.val);
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
        return res;
    }

}
