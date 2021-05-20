package offer;

// 37 序列化二叉树
class Codec {

  public String serialize(TreeNode root) {
    StringBuilder res = new StringBuilder();
    serializeHelper(root, res);
    return res.toString();
  }

  private void serializeHelper(TreeNode root, StringBuilder sb) {
    if (root == null) {
      sb.append("#");
      return;
    }
    sb.append("(");
    serializeHelper(root.left, sb);
    sb.append(")");
    sb.append(root.val); // 中序遍历
    sb.append("(");
    serializeHelper(root.right, sb);
    sb.append(")");
  }

  public TreeNode deserialize(String data) {
    int[] ptr = {0};
    return parse(data, ptr);
  }

  public TreeNode parse(String data, int[] ptr) {
    if (data.charAt(ptr[0]) == '#') {
      ++ptr[0];
      return null;
    }
    TreeNode cur = new TreeNode(0);
    cur.left = parseSubtree(data, ptr);
    cur.val = parseInt(data, ptr);
    cur.right = parseSubtree(data, ptr);
    return cur;
  }

  public TreeNode parseSubtree(String data, int[] ptr) {
    ++ptr[0]; // 跳过左括号
    TreeNode subtree = parse(data, ptr);
    ++ptr[0]; // 跳过右括号
    return subtree;
  }

  public int parseInt(String data, int[] ptr) {
    int x = 0, sgn = 1;
    if (!Character.isDigit(data.charAt(ptr[0]))) {
      sgn = -1;
      ++ptr[0];
    }
    while (Character.isDigit(data.charAt(ptr[0]))) {
      x = x * 10 + data.charAt(ptr[0]++) - '0';
    }
    return x * sgn;
  }
}
