package algo;

class TreeNode {

  int val;
  TreeNode left, right;

  TreeNode() {
  }

  TreeNode(int val, TreeNode left, TreeNode right) {
    this.val = val;
    this.left = left;
    this.right = right;
  }

  void traverse(TreeNode root) {
    if (root == null) {
      return;
    }
    // pre
    traverse(root.left);
    //
    traverse(root.right);
    // post
  }
}

