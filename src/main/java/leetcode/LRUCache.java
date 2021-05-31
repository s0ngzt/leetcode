package leetcode;

import java.util.HashMap;
import java.util.Map;

// 146 LRU 缓存
// TODO
class LRUCache {

  static class DuLinkedNode {

    int key, value;
    DuLinkedNode prev, next;

    public DuLinkedNode() {
    }

    public DuLinkedNode(int _key, int _value) {
      key = _key;
      value = _value;
    }
  }

  private final Map<Integer, DuLinkedNode> cache = new HashMap<>();
  private int size;
  private final int capacity;
  private final DuLinkedNode head;
  private final DuLinkedNode tail;

  public LRUCache(int capacity) {
    this.size = 0;
    this.capacity = capacity;
    head = new DuLinkedNode();
    tail = new DuLinkedNode();
    head.next = tail;
    tail.prev = head;
  }

  public int get(int key) {
    DuLinkedNode node = cache.get(key);
    if (node == null) {
      return -1;
    }
    // 如果 key 存在，先通过哈希表定位，再移到头部
    moveToHead(node);
    return node.value;
  }

  public void put(int key, int value) {
    DuLinkedNode node = cache.get(key);
    if (node == null) {
      // 如果 key 不存在，创建一个新的节点
      DuLinkedNode newNode = new DuLinkedNode(key, value);
      // 添加进哈希表
      cache.put(key, newNode);
      // 添加至双向链表的头部
      addToHead(newNode);
      ++size;
      if (size > capacity) {
        // 如果超出容量，删除双向链表的尾部节点
        DuLinkedNode tail = removeTail();
        // 删除哈希表中对应的项
        cache.remove(tail.key);
        --size;
      }
    } else {
      // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
      node.value = value;
      moveToHead(node);
    }
  }

  private void addToHead(DuLinkedNode node) {
    node.prev = head;
    node.next = head.next;
    head.next.prev = node;
    head.next = node;
  }

  private void removeNode(DuLinkedNode node) {
    node.prev.next = node.next;
    node.next.prev = node.prev;
  }

  private void moveToHead(DuLinkedNode node) {
    removeNode(node);
    addToHead(node);
  }

  private DuLinkedNode removeTail() {
    DuLinkedNode res = tail.prev;
    removeNode(res);
    return res;
  }
}