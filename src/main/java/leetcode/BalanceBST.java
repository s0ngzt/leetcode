package leetcode;

import java.util.ArrayList;
import java.util.List;

/**
 * 给一棵二叉搜索树 返回一棵 "平衡" 后的二叉搜索树 新生成的树应与原来的树有相同的节点值
 */
class BalanceBST {

    List<Integer> inorderArray = new ArrayList<>();

    public TreeNode balanceBST(TreeNode root) {
        getInorder(root);
        return build(0, inorderArray.size() - 1);
    }

    public void getInorder(TreeNode o) {
        if (o.left != null) {
            getInorder(o.left);
        }
        inorderArray.add(o.val);
        if (o.right != null) {
            getInorder(o.right);
        }
    }

    public TreeNode build(int l, int r) {
        int mid = l + (r - l) / 2;
        TreeNode o = new TreeNode(inorderArray.get(mid));
        if (l <= mid - 1) {
            o.left = build(l, mid - 1);
        }
        if (mid + 1 <= r) {
            o.right = build(mid + 1, r);
        }
        return o;
    }
}
