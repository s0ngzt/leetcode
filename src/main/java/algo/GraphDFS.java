package algo;

public class GraphDFS {

  private boolean[] marked;
  private int count;

  public GraphDFS(Graph g, int s) {
    marked = new boolean[g.getNumVertex()];
    dfs(g, s);
  }

  private void dfs(Graph g, int v) {
    marked[v] = true;
    for (var w : g.adj(v)) {
      if (!marked[w]) {
        dfs(g, w);
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
