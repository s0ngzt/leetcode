package leetcode;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

class NodeSolution {

  // 1689
  public int minPartitions(String n) {
    int min = 0;
    for (var c : n.toCharArray()) {
      if (c - '0' > min) {
        min = c - '0';
      }
    }
    return min;
  }

  // 429
  public List<List<Integer>> levelOrder(Node root) {
    List<List<Integer>> result = new LinkedList<>();
    if (root == null) {
      return result;
    }
    Queue<Node> queue = new LinkedList<>();
    queue.add(root);
    while (!queue.isEmpty()) {
      List<Integer> level = new LinkedList<>();
      int size = queue.size();
      for (int i = 0; i < size; i++) {
        Node node = queue.poll();
        assert node != null;
        level.add(node.val);
        queue.addAll(node.children);
      }
      result.add(level);
    }
    return result;
  }

  // 590
  public List<Integer> postorder(Node root) {
    LinkedList<Node> stack = new LinkedList<>();
    LinkedList<Integer> out = new LinkedList<>();
    if (root == null) {
      return out;
    }
    stack.add(root);
    while (!stack.isEmpty()) {
      Node node = stack.pollLast();
      out.addFirst(node.val);
      for (Node item : node.children) {
        if (item != null) {
          stack.add(item);
        }
      }
    }
    return out;
  }

  // 589
  public List<Integer> preorder(Node root) {
    var res = new LinkedList<Integer>();
    if (root == null) {
      return res;
    }
    var stack = new LinkedList<Node>();
    stack.add(root);
    while (!stack.isEmpty()) {
      var node = stack.pollLast();
      res.add(node.val);
      Collections.reverse(node.children);
      stack.addAll(node.children);
    }
    return res;
  }

}
