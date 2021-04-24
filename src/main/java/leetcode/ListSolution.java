package leetcode;

class ListSolution {

  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) {
      return null;
    }
    ListNode pA = headA, pB = headB;
    // ListA.last.next = ListB.head
    // ListB.last.next = ListA.head
    while (pA != pB) {
      if (pA == null) {
        pA = headB;
      } else {
        pA = pA.next;
      }
      if (pB == null) {
        pB = headA;
      } else {
        pB = pB.next;
      }
    }
    return pA;
  }


  public void deleteNode(ListNode node) {
    node.val = node.next.val;
    node.next = node.next.next;
  }


  // 206
  public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }
    ListNode last = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return last;
  }

  public ListNode reverseBetween(ListNode head, int left, int right) {

    if (head == null || left >= right) {
      return head;
    }
    ListNode newHead = new ListNode(0, head);
    ListNode pre = newHead;
    for (int count = 0; pre.next != null && count < left - 1; count++) {
      pre = pre.next;
    }
    if (pre.next == null) {
      return head;
    }
    var cur = pre.next;
    for (int i = 0; i < right - left; i++) {
      var tmp = pre.next;
      pre.next = cur.next;
      cur.next = cur.next.next;
      pre.next.next = tmp;
    }
    return newHead.next;
  }

  // 61
  public ListNode rotateRight(ListNode head, int k) {
    if (k == 0 || head == null || head.next == null) {
      return head;
    }
    int listLength = 1;
    ListNode cursor = head;  // 游标
    while (cursor.next != null) {
      cursor = cursor.next;
      listLength++;
    }
    int add = listLength - k % listLength;
    if (add == listLength) {
      return head;
    }
    cursor.next = head; // 形成 环
    while (add-- > 0) {
      // 顺时针 移动 add 次
      cursor = cursor.next;
    }
    ListNode newHead = cursor.next;
    cursor.next = null;
    return newHead;
  }

  // 83
  public ListNode deleteDuplicates(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }

    ListNode cursor = head;
    while (cursor.next != null) {
      if (cursor.val == cursor.next.val) {
        int x = cursor.val;
        while (cursor.next != null && cursor.next.val == x) {
          cursor.next = cursor.next.next;
        }
      } else {
        cursor = cursor.next;
      }
    }

    return head;
  }

  public ListNode deleteAllDuplicates(ListNode head) {

    if (head == null || head.next == null) {
      return head;
    }

    ListNode fakeHead = new ListNode(0, head);

    ListNode cursor = fakeHead;
    while (cursor.next != null && cursor.next.next != null) {
      if (cursor.next.val == cursor.next.next.val) {
        int x = cursor.next.val;
        while (cursor.next != null && cursor.next.val == x) {
          cursor.next = cursor.next.next;
        }
      } else {
        cursor = cursor.next;
      }
    }

    return fakeHead.next;
  }
}
