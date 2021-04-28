package leetcode;

class ListSolution {

  // 24 medium 两两交换链表中的节点
  public ListNode swapPairs(ListNode head) {
    return swapPairsHelper(head);
  }

  private ListNode swapPairsHelper(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }
    // 保存下一阶段的头指针
    var nextHead = head.next.next;
    // 翻转当前阶段指针
    var next = head.next;
    next.next = head;
    head.next = swapPairsHelper(nextHead);
    return next;
  }

  // 234 easy 回文链表
  public boolean isPalindrome(ListNode head) {
    // 1 2 1 nil
    // 1 2 2 1 nil
    if (head == null) {
      return true;
    }
    var slow = head;
    // fast 如果初始化为 head.Next, 则中点在 slow.Next
    // fast 初始化为 head, 则中点在 slow
    var fast = head.next;
    while (fast != null && fast.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }

    var tail = reverseList(slow.next);
    // 断开两个链表(需要用到中点前一个节点)
    slow.next = null;
    while (head != null && tail != null) {
      if (head.val != tail.val) {
        return false;
      }
      head = head.next;
      tail = tail.next;
    }
    return true;
  }

  // 142 medium 环形链表 II
  // 快慢指针，快慢相遇之后，其中一个指针回到头，快慢指针步调一致一起移动，相遇点即为入环点
  public ListNode detectCycle(ListNode head) {
    if (head == null || head.next == null) {
      return null;
    }
    var slow = head;
    var fast = head.next;
    while (fast != null && fast.next != null) {
      if (fast == slow) {
        fast = head;
        slow = slow.next;
        while (fast != slow) {
          fast = fast.next;
          slow = slow.next;
        }
        return slow;
      }
      slow = slow.next;
      fast = fast.next.next;
    }
    return null;
  }

  // 141 easy 环形链表
  public boolean hasCycle(ListNode head) {
    if (head == null || head.next == null) {
      return false;
    }
    var slow = head;
    var fast = head.next;
    while (slow != fast) {
      if (fast == null || fast.next == null) {
        return false;
      }
      slow = slow.next;
      fast = fast.next.next;
    }
    return true;
  }

  // 143 medium 重排链表
  public void reorderList(ListNode head) {
    if (head == null) {
      return;
    }
    var mid = findMiddle(head);
    var tail = reverseList(mid.next);
    mid.next = null;
    head = mergeTwoListsInTurn(head, tail);
  }

  private ListNode mergeTwoListsInTurn(ListNode l1, ListNode l2) {
    var dummyHead = new ListNode(0);
    var cursor = dummyHead;
    var toggle = true;
    while (l1 != null && l2 != null) {
      if (toggle) {
        cursor.next = l1;
        l1 = l1.next;
      } else {
        cursor.next = l2;
        l2 = l2.next;
      }
      toggle = !toggle;
      cursor = cursor.next;
    }

    while (l1 != null) {
      cursor.next = l1;
      cursor = cursor.next;
      l1 = l1.next;
    }
    while (l2 != null) {
      cursor.next = l2;
      cursor = cursor.next;
      l2 = l2.next;
    }
    return dummyHead.next;
  }

  // 148 medium 排序链表
  public ListNode sortList(ListNode head) {
    return mergeSort(head);
  }

  private ListNode mergeSort(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }

    var middle = findMiddle(head);
    var tail = middle.next;
    middle.next = null;

    var left = mergeSort(head);
    var right = mergeSort(tail);
    return mergeTwoLists(left, right);
  }

  private ListNode findMiddle(ListNode head) {
    var slow = head;
    var fast = head.next;
    while (fast != null && fast.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }
    return slow;
  }

  // 86 medium 分割链表
  public ListNode partition(ListNode head, int x) {
    if (head == null) {
      return null;
    }

    var dummyHead = new ListNode(0);
    var dummyTail = new ListNode(0);

    var tail = dummyTail;
    dummyHead.next = head;
    head = dummyHead;
    while (head.next != null) {
      if (head.next.val < x) {
        head = head.next;
      } else {
        var tmp = head.next;
        head.next = head.next.next;
        tail.next = tmp;
        tail = tail.next;
      }
    }
    tail.next = null;
    head.next = dummyTail.next;
    return dummyHead.next;
  }

  // 21 easy 合并两个有序链表
  public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    var dummyHead = new ListNode(0);
    var cursor = dummyHead;
    while (l1 != null && l2 != null) {
      if (l1.val < l2.val) {
        cursor.next = l1;
        l1 = l1.next;
      } else {
        cursor.next = l2;
        l2 = l2.next;
      }
      cursor = cursor.next;
    }

    while (l1 != null) {
      cursor.next = l1;
      cursor = cursor.next;
      l1 = l1.next;
    }
    while (l2 != null) {
      cursor.next = l2;
      cursor = cursor.next;
      l2 = l2.next;
    }
    return dummyHead.next;
  }

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


  // 206 easy 反转链表
  public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }
    ListNode last = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return last;
  }

  // 92 medium 反转链表 II
  public ListNode reverseBetween(ListNode head, int left, int right) {

    if (head == null || left >= right) {
      return head;
    }
    ListNode newHead = new ListNode(0, head);
    ListNode pre = newHead;
    // find left
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

  // 83 easy 删除排序链表中的重复元素
  public ListNode deleteDuplicates(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }

    ListNode current = head;
    while (current.next != null) {
      if (current.val == current.next.val) {
        while (current.next != null && current.next.val == current.val) {
          current.next = current.next.next;
        }
      } else {
        current = current.next;
      }
    }

    return head;
  }

  // 82 medium 删除排序链表中的重复元素 II
  public ListNode deleteAllDuplicates(ListNode head) {

    if (head == null || head.next == null) {
      return head;
    }

    ListNode dummyHead = new ListNode(0, head);

    ListNode cursor = dummyHead;
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

    return dummyHead.next;
  }
}
