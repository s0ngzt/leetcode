package offer;

import java.util.Deque;
import java.util.LinkedList;

class CQueue {

    Deque<Integer> inStack;
    Deque<Integer> outStack;

    public CQueue() {
        inStack = new LinkedList<>();
        outStack = new LinkedList<>();
    }

    public void appendTail(int value) {
        inStack.push(value);
    }

    public int deleteHead() {
        if (isEmpty()) {
            return -1;
        }
        move();
        return outStack.pollFirst();
    }

    private boolean isEmpty() {
        return inStack.isEmpty() && outStack.isEmpty();
    }

    private void move() {
        if (outStack.isEmpty()) {
            while (!inStack.isEmpty()) {
                outStack.offerFirst(inStack.pollFirst());
            }
        }
    }
}