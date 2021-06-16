package algo;

class ListNode {

    int val;
    ListNode next;

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

    // 环入口
    public static ListNode detectCycle(ListNode head) {
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
}
