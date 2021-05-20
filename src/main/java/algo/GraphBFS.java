package algo;

import java.util.Deque;
import java.util.LinkedList;

public class GraphBFS {

  private boolean[] marked;
  private int count;
  private Deque<Integer> nodeToSearch;

  public GraphBFS(Graph g, int s) {
    marked = new boolean[g.getNumVertex()];
    nodeToSearch = new LinkedList<>();
    bfs(g, s);
  }

  private void bfs(Graph g, int v) {
    marked[v] = true;
    nodeToSearch.offer(v);
    while (!nodeToSearch.isEmpty()) {
      var n = nodeToSearch.poll();

      for (var w : g.adj(n)) {
        if (!marked[w]) {
          bfs(g, w);
        }
      }
    }
    count++;
  }

  // 判断 w 顶点与 s 顶点是否相通
  public boolean marked(int w) {
    return marked[w];
  }

  // 获取与顶点 s 相通的所有顶点的总数
  public int count() {
    return count;
  }
}
