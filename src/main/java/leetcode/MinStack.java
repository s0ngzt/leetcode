package leetcode;

import java.util.Deque;
import java.util.LinkedList;

class MinStack {

  Deque<Integer> numStack;
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

  public int getMin() {
    assert (!minStack.isEmpty());
    return minStack.peek();
  }
}