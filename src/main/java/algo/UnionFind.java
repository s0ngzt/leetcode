package algo;

public class UnionFind {

  private int[] elemAndGroup;
  private int count;

  public UnionFind(int n) {
    this.count = n;
    elemAndGroup = new int[n];
    for (int i = 0; i < n; i++) {
      elemAndGroup[i] = i;
    }
  }

  public int getCount() {
    return count;
  }

  // 元素 p 所在分组
  public int find(int p) {
    return elemAndGroup[p];
  }

  public void union(int p, int q) {
    // 已经在同一分组中
    if (inSameGroup(p, q)) {
      return;
    }

    int groupOfP = find(p);
    int groupOfQ = find(q);

    // 合并组（让 p 所在组的所有元素的组标识符变为 q 所在分组的标识符）
    for (int i = 0; i < elemAndGroup.length; i++) {
      if (elemAndGroup[i] == groupOfP) {
        elemAndGroup[i] = groupOfQ;
      }
    }

    this.count--;
  }

  // 判断并查集中元素 p 和元素 q 是否在同一分组中
  public boolean inSameGroup(int p, int q) {
    return find(p) == find(q);
  }
}
