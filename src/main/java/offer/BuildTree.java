package offer;

import java.util.HashMap;
import java.util.Map;

// 07. 重建二叉树
public class BuildTree {

  private Map<Integer, Integer> indexMap;

  public TreeNode buildTree(int[] preorder, int[] inorder) {
    int n = preorder.length;
    indexMap = new HashMap<>();
    for (int i = 0; i < n; i++) {
      indexMap.put(inorder[i], i);
    }
    return buildTree(preorder, inorder, 0, n - 1, 0, n - 1);
  }

  public TreeNode buildTree(int[] preorder, int[] inorder, int preorderLeft, int preorderRight,
      int inorderLeft, int inorderRight) {
    if (preorderLeft > preorderRight) {
      return null;
    }

    int inorderRoot = indexMap.get(preorder[preorderLeft]);

    // 先把根节点建立出来
    TreeNode root = new TreeNode(preorder[preorderLeft]);
    // 得到左子树中的节点数目
    int sizeLeftSubtree = inorderRoot - inorderLeft;
    // 递归地构造左子树，并连接到根节点
    root.left = buildTree(preorder, inorder, preorderLeft + 1, preorderLeft + sizeLeftSubtree,
        inorderLeft, inorderRoot - 1);
    // 递归地构造右子树，并连接到根节点
    root.right = buildTree(preorder, inorder, preorderLeft + sizeLeftSubtree + 1,
        preorderRight, inorderRoot + 1, inorderRight);
    return root;
  }
}
