package offer;

import java.util.Deque;
import java.util.LinkedList;

// 30 包含 min() 的栈
class MinStack {

  Deque<Integer> numStack;
  // 维护<最小值>栈
  Deque<Integer> minStack;

  public MinStack() {
    numStack = new LinkedList<>();
    minStack = new LinkedList<>();
    minStack.push(Integer.MAX_VALUE);
  }

  public void push(int x) {
    numStack.push(x);
    assert (!minStack.isEmpty());
    minStack.push(Math.min(minStack.peek(), x));
  }

  public void pop() {
    numStack.pop();
    minStack.pop();
  }

  public int top() {
    assert (!numStack.isEmpty());
    return numStack.peek();
  }

  public int min() {
    assert (!minStack.isEmpty());
    return minStack.peek();
  }
}