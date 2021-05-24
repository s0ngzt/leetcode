package interview;

import java.util.Deque;
import java.util.LinkedList;

// 面试题 03.05. 栈排序
class SortedStack {

  private final Deque<Integer> sortedStack;
  private final Deque<Integer> helperStack;

  public SortedStack() {
    sortedStack = new LinkedList<>();
    helperStack = new LinkedList<>();
  }

  public void push(int val) {
    if (sortedStack.isEmpty()) {
      sortedStack.push(val);
    } else {
      while (!sortedStack.isEmpty() && sortedStack.peek() < val) {
        helperStack.push(sortedStack.pop());
      }
      sortedStack.push(val);
      while (!helperStack.isEmpty()) {
        sortedStack.push(helperStack.pop());
      }
    }
  }

  public void pop() {
    if (!sortedStack.isEmpty()) {
      sortedStack.pop();
    }
  }

  public int peek() {
    if (sortedStack.isEmpty()) {
      return -1;
    }
    return sortedStack.peek();
  }

  public boolean isEmpty() {
    return sortedStack.isEmpty();
  }
}
