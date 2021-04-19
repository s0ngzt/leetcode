package interview;

import java.util.LinkedList;

class StackOfPlates {

  private LinkedList<LinkedList<Integer>> stacks;
  private int capacity;

  public StackOfPlates(int cap) {
    this.stacks = new LinkedList<>();
    this.capacity = cap;
  }

  private boolean setIsEmpty() {
    return stacks.isEmpty();
  }

  private boolean lastStackIsFUll() {
    if (setIsEmpty()) {
      return true;
    }
    return stacks.getLast().size() >= capacity;
  }

  private boolean lastStackIsEmpty() {
    return stacks.getLast().isEmpty();
  }

  public void push(int val) {
    if (capacity <= 0) {
      return;
    }
    if (setIsEmpty() || lastStackIsFUll()) {
      stacks.addLast(new LinkedList<>());
    }
    stacks.getLast().addLast(val);
  }

  public int pop() {
    int val = -1;
    if (setIsEmpty()) {
      return val;
    }
    val = stacks.getLast().removeLast();
    if (lastStackIsEmpty()) {
      stacks.removeLast();
    }
    return val;
  }

  public int popAt(int index) {
    int val = -1;
    if (setIsEmpty() || stacks.size() - 1 < index) {
      return val;
    }
    val = stacks.get(index).removeLast();
    if (stacks.get(index).isEmpty()) {
      stacks.remove(index);
    }
    return val;
  }
}

