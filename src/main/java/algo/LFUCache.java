package algo;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * LFU
 * <p>
 * Least Frequently Used
 * <p>
 * 最不经常使用
 * <p>
 * LFU 更新和插入新页面可以发生在链表中任意位置，删除页面都发生在表尾
 */
class LFUCache {

  static class Node {

    int key, val, freq;

    Node(int key, int val, int freq) {
      this.key = key;
      this.val = val;
      this.freq = freq;
    }
  }

  int minFreq, capacity;
  Map<Integer, LinkedList<Node>> freq_table; // 频率作为索引, 双端队列存放存放所有使用频率为 freq 的缓存
  Map<Integer, Node> key_table;

  public LFUCache(int capacity) {
    this.minFreq = 0;
    this.capacity = capacity;
    freq_table = new HashMap<>();
    key_table = new HashMap<>();
  }

  public int get(int key) {
    if (capacity == 0) {
      return -1;
    }
    if (!key_table.containsKey(key)) {
      return -1;
    }
    Node node = key_table.get(key);
    int val = node.val, freq = node.freq;
    freq_table.get(freq).remove(node);
    // 如果当前链表为空，需要在哈希表中删除，且更新 minFreq
    if (freq_table.get(freq).size() == 0) {
      freq_table.remove(freq);
      if (minFreq == freq) {
        minFreq += 1;
      }
    }
    // 插入到 freq + 1 中
    LinkedList<Node> list = freq_table.getOrDefault(freq + 1, new LinkedList<>());
    list.offerFirst(new Node(key, val, freq + 1));
    freq_table.put(freq + 1, list);
    key_table.put(key, freq_table.get(freq + 1).peekFirst());
    return val;
  }

  public void put(int key, int value) {
    if (capacity == 0) {
      return;
    }
    if (!key_table.containsKey(key)) {
      // 缓存已满，需要进行删除操作
      if (key_table.size() == capacity) {
        // 通过 minFreq 拿到 freq_table[minFreq] 链表的末尾节点
        Node node = freq_table.get(minFreq).peekLast();
        assert node != null;
        key_table.remove(node.key);
        freq_table.get(minFreq).pollLast();
        if (freq_table.get(minFreq).size() == 0) {
          freq_table.remove(minFreq);
        }
      }
      LinkedList<Node> list = freq_table.getOrDefault(1, new LinkedList<>());
      list.offerFirst(new Node(key, value, 1));
      freq_table.put(1, list);
      key_table.put(key, freq_table.get(1).peekFirst());
      minFreq = 1;
    } else {
      // 与 get 操作基本一致，除了需要更新缓存的值
      Node node = key_table.get(key);
      int freq = node.freq;
      freq_table.get(freq).remove(node);
      if (freq_table.get(freq).size() == 0) {
        freq_table.remove(freq);
        if (minFreq == freq) {
          minFreq += 1;
        }
      }
      LinkedList<Node> list = freq_table.getOrDefault(freq + 1, new LinkedList<>());
      list.offerFirst(new Node(key, value, freq + 1));
      freq_table.put(freq + 1, list);
      key_table.put(key, freq_table.get(freq + 1).peekFirst());
    }
  }
}
