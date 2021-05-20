package algo;

import java.util.Optional;

public class BinarySearchTree<K extends Comparable<? super K>, V> {

  private static class Node<K, V> {

    K key;
    V value;
    Node<K, V> left, right;

    Node() {
    }

    Node(K k, V v) {
      this.key = k;
      this.value = v;
    }

    Node(K k, V v, Node<K, V> left, Node<K, V> right) {
      this.key = k;
      this.value = v;
      this.left = left;
      this.right = right;
    }

  }

  private Node<K, V> root;

  public BinarySearchTree() {
    root = null;
  }

  public void makeEmpty() {
    root = null;
  }

  public boolean isEmpty() {
    return root == null;
  }

  public Optional<V> get(K key) {
    return get(root, key);
  }

  private Optional<V> get(Node<K, V> root, K key) {
    if (root == null) {
      return Optional.empty();
    }
    int comparisonResult = key.compareTo(root.key);
    if (comparisonResult == 0) {
      return Optional.of(root.value);
    } else if (comparisonResult < 0) {
      return get(root.left, key);
    } else {
      return get(root.right, key);
    }
  }

  public Optional<K> getMinKey() {
    return getMin(root).map(node -> node.key);
  }

  private Optional<Node<K, V>> getMin(Node<K, V> root) {
    if (root == null) {
      return Optional.empty();
    }
    if (root.left == null) {
      return Optional.of(root);
    }
    return getMin(root.left);
  }

  public Optional<K> getMaxKey() {
    return getMax(root).map(node -> node.key);
  }

  private Optional<Node<K, V>> getMax(Node<K, V> root) {
    if (root == null) {
      return Optional.empty();
    }
    if (root.right == null) {
      return Optional.of(root);
    }
    return getMax(root.right);
  }

  public void insert(K key, V value) {
    root = insert(root, key, value);
  }

  private Node<K, V> insert(Node<K, V> root, K key, V value) {
    if (root == null) {
      root = new Node<>(key, value);
      return root;
    }
    int comparisonResult = key.compareTo(root.key);
    if (comparisonResult == 0) {
      root.value = value;
    } else if (comparisonResult < 0) {
      root.left = insert(root.left, key, value);
    } else {
      root.right = insert(root.right, key, value);
    }
    return root;
  }

  public void delete(K key) {
    deleteBST(root, key);
  }

  private Node<K, V> deleteBST(Node<K, V> root, K key) {
    if (root == null) {
      return null;
    }
    int comparableResult = key.compareTo(root.key);
    if (comparableResult < 0) {
      root.left = deleteBST(root.left, key);
    } else if (comparableResult > 0) {
      root.right = deleteBST(root.right, key);
    } else {
      root = deleteNode(root);
    }
    return root;
  }

  private Node<K, V> deleteNode(Node<K, V> root) {
    if (root.left == null && root.right == null) {
      return null;
    }
    if (root.right == null) {
      return root.left;
    } else if (root.left == null) {
      return root.right;
    } else {
      var q = root;
      var s = root.left;
      while (s.right != null) {
        q = s;
        s = s.right;
      }
      // s 为 root 的前驱
      root.key = s.key;
      root.value = s.value;
      if (q != root) {
        q.right = s.left;
      } else {
        q.left = s.left;
      }
    }

    return root;
  }

  public void print() {
    printTree(root);
    System.out.println();
  }

  private void printTree(Node<K, V> t) {
    if (t != null) {
      printTree(t.left);
      System.out.print(t.key);
      System.out.print(" ");
      printTree(t.right);
    }
  }
}
