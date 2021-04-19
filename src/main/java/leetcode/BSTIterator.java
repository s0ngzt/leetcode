package leetcode;

import java.util.ArrayDeque;
import java.util.Deque;

class BSTIterator {

  Deque<TreeNode> d = new ArrayDeque<>();

  public BSTIterator(TreeNode root) {
    dfsLeft(root);
  }

  public int next() {
    TreeNode root = d.pollLast();
    assert root != null;
    int ans = root.val;
    root = root.right;
    dfsLeft(root);
    return ans;
  }

  void dfsLeft(TreeNode root) {
    while (root != null) {
      d.addLast(root);
      root = root.left;
    }
  }

  public boolean hasNext() {
    return !d.isEmpty();
  }
}