package leetcode;

import java.util.HashMap;
import java.util.Map;

class NodeSolution {

  public Node cloneGraph(Node node) {
    Map<Node, Node> visited = new HashMap<>();
    return cloning(node, visited);

  }

  private Node cloning(Node node, Map<Node, Node> visited) {
    if (node == null) {
      return null;
    }
    // 已经访问过直接返回
    if (visited.containsKey(node)) {
      return visited.get(node);
    }
    var newNode = new Node(node.val);
    visited.put(node, newNode);
    for (int i = 0; i < node.neighbors.size(); i++) {
      newNode.neighbors.add(cloning(node.neighbors.get(i), visited));
    }
    return newNode;
  }

}
