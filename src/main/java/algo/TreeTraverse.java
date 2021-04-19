package algo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

public class TreeTraverse {

  /**
   * 先序遍历
   *
   * @param root 根节点
   * @return 遍历结果
   */
  public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if (root == null) {
      return res;
    }
    Deque<TreeNode> stack = new LinkedList<>();
    TreeNode node = root;
    while (!stack.isEmpty() || node != null) {
      while (node != null) {
        res.add(node.val);
        stack.push(node);
        node = node.left;
      }
      node = stack.pop();
      node = node.right;
    }
    return res;
  }

  /**
   * 中序遍历
   *
   * @param root 根节点
   * @return 遍历结果
   */
  public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    Stack<TreeNode> stack = new Stack<>();
    while (root != null || !stack.isEmpty()) {
      while (root != null) {
        stack.push(root);
        root = root.left;
      }
      root = stack.pop();
      res.add(root.val);
      root = root.right;
    }
    return res;
  }

  /**
   * 后序遍历
   *
   * @param root 根节点
   * @return 遍历结果
   */
  public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    Deque<TreeNode> stack = new LinkedList<>();

    while (root != null || !stack.isEmpty()) {
      while (root != null) {
        res.add(root.val);
        stack.push(root);
        root = root.right;
      }

      TreeNode cur = stack.pop();
      root = cur.left;
    }

    Collections.reverse(res);
    return res;
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
