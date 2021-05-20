package algo;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

public class Graph {

  private int numVertex;
  private int numEdge;

  protected List<Deque<Integer>> adj;

  public Graph(int v) {
    this.numVertex = v;
    this.numEdge = 0;
    this.adj = new ArrayList<>(v);
    for (int i = 0; i < v; i++) {
      adj.set(i, new LinkedList<>());
    }
  }

  public int getNumVertex() {
    return numVertex;
  }

  public int getNumEdge() {
    return numEdge;
  }

  public void addEdge(int v, int w) {
    adj.get(v).offer(w);
    adj.get(w).offer(v);
    numEdge += 1;
  }

  public Deque<Integer> adj(int v) {
    return adj.get(v);
  }

}
