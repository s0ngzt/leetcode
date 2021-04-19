package algo;

class ListNode {

  int val;
  ListNode next;

  void traverse(ListNode head) {
    for (ListNode p = head; p != null; p = p.next) {
      // TODO
      System.out.println(p.val);
    }
  }

  void traverseRecursively(ListNode head) {
    // TODO
    traverse(head.next);
  }

  /**
   * 递归反转整个链表
   */
  ListNode reverse(ListNode head) {
    if (head.next == null) {
      return head;
    }
    ListNode last = reverse(head.next);
    head.next.next = head;
    head.next = null;
    return last;
  }
}
