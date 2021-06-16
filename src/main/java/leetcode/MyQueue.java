package leetcode;

import java.util.Stack;

/**
 * 双栈 实现 队列
 */
class MyQueue {

    Stack<Integer> in;
    Stack<Integer> out;

    public MyQueue() {
        in = new Stack<>();
        out = new Stack<>();
    }

    public void push(int x) {
        in.push(x);
    }

    public int pop() {
        if (out.isEmpty()) {
            in2out();
        }
        return out.pop();
    }

    public int peek() {
        if (out.isEmpty()) {
            in2out();
        }
        return out.peek();
    }

    public boolean empty() {
        return in.empty() && out.empty();
    }

    private void in2out() {
        while (!in.isEmpty()) {
            out.push(in.pop());
        }
    }

}
