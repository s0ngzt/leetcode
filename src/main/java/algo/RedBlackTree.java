package algo;

public class RedBlackTree<Key extends Comparable<? super Key>, Value> {

  // 根节点
  private Node root;
  // 记录树中元素的个数
  private int N;
  // 红色
  private static final boolean RED = true;
  // 黑色
  private static final boolean BLACK = false;

  // 结点类
  private class Node {

    public Key key;
    private Value value;
    public Node left;
    public Node right;
    public boolean color;

    public Node(Key key, Value value, Node left, Node right, boolean color) {
      this.key = key;
      this.value = value;
      this.left = left;
      this.right = right;
      this.color = color;
    }
  }

  // 获取树中元素的个数
  public int size() {
    return N;
  }

  private boolean isRed(Node x) {
    if (x == null) {
      return false;
    }
    return x.color;
  }

  private boolean isBlack(Node x) {
    return !isRed(x);
  }

  /**
   * 左旋
   *
   * @param n 节点
   * @return 左旋后
   */
  private Node rotateLeft(Node n) {
    // 找到 n 结点的右子结点 r
    Node r = n.right;
    n.right = r.left;
    r.left = n;
    //让x结点的color属性变为h结点的color属性
    r.color = n.color;
    //让 h 结点的 color 属性变为 RED
    n.color = RED;
    return r;
  }

  /**
   * 右旋
   *
   * @param n 节点
   * @return 右旋后
   */
  private Node rotateRight(Node n) {
    //找到 n 结点的左子结点 l
    Node l = n.left;
    n.left = l.right;
    l.right = n;
    l.color = n.color;
    n.color = RED;

    return l;
  }

  /**
   * 在整个树上完成插入操作
   *
   * @param key 键
   * @param val 值
   */
  public void put(Key key, Value val) {
    root = put(root, key, val);
    // 根结点的颜色总是黑色
    root.color = BLACK;
  }

  private Node put(Node h, Key key, Value val) {
    // 判断 h 是否为空，如果为空则直接返回一个红色的结点
    if (h == null) {
      N++;
      return new Node(key, val, null, null, RED);
    }

    // 比较 h 结点的键和 key 的大小
    int cmp = key.compareTo(h.key);
    if (cmp < 0) {
      // 继续往左
      h.left = put(h.left, key, val);

    } else if (cmp > 0) {
      // 继续往右
      h.right = put(h.right, key, val);

    } else {
      // 发生值的替换
      h.value = val;
    }

    // 左旋（当前结点 h 的左子结点为黑色，右子结点为红色，需要左旋）
    if (isBlack(h.left) && isRed(h.right)) {
      h = rotateLeft(h);
    }

    // 右旋（当前结点 h 的左子结点和左子结点的左子结点都为红色，需要右旋）
    if (isRed(h.left) && isRed(h.left.left)) {
      rotateRight(h);
    }

    // 颜色反转（当前结点的左子结点和右子结点都为红色时，需要颜色反转）
    if (isRed(h.left) && isRed(h.right)) {
      flipColors(h);
    }

    return h;
  }

  /**
   * 颜色反转
   */
  private void flipColors(Node h) {
    // 当前结点变为红色
    h.color = RED;
    // 左子结点和右子结点变为黑色
    h.left.color = BLACK;
    h.right.color = BLACK;
  }

  // 根据 key，查找 value
  public Value get(Key key) {
    return get(root, key);
  }

  private Value get(Node x, Key key) {
    if (x == null) {
      return null;
    }
    int cmp = key.compareTo(x.key);
    if (cmp < 0) {
      return get(x.left, key);
    } else if (cmp > 0) {
      return get(x.right, key);
    } else {
      return x.value;
    }
  }
}
