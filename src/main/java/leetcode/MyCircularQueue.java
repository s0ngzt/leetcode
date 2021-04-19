package leetcode;

class MyCircularQueue {

  int[] buffer;
  int front, rear, capacity, size;

  public MyCircularQueue(int k) {
    buffer = new int[k];
    front = 0;
    rear = 0;
    size = 0;
    capacity = k;
  }

  public boolean enQueue(int value) {
    if (size == capacity) {
      return false;
    }
    size++;
    buffer[rear] = value;
    rear = (rear + 1) % capacity;
    return true;
  }

  public boolean deQueue() {
    if (size == 0) {
      return false;
    }
    size--;
    front = (front + 1) % capacity;
    return true;
  }

  public int Front() {
    if (size == 0) {
      return -1;
    }
    return buffer[front];
  }

  public int Rear() {
    if (size == 0) {
      return -1;
    }
    if (rear == 0) {
      return buffer[capacity - 1];
    }
    return buffer[this.rear - 1];
  }

  public boolean isEmpty() {
    return size == 0;
  }

  public boolean isFull() {
    return size == capacity;
  }
}
