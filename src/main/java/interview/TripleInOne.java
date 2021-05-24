package interview;

// 面试题 03.01. 三合一
class TripleInOne {

  int N = 3;
  int[] data;
  int[] locations;
  int size;

  public TripleInOne(int stackSize) {
    size = stackSize;
    data = new int[size * N];
    locations = new int[N];
    for (int i = 0; i < N; i++) {
      locations[i] = i * size;
    }
  }

  public void push(int stackNum, int value) {
    int idx = locations[stackNum];
    if (idx < (stackNum + 1) * size) {
      data[idx] = value;
      locations[stackNum]++;
    }
  }

  public int pop(int stackNum) {
    int idx = locations[stackNum];
    if (idx > stackNum * size) {
      locations[stackNum]--;
      return data[idx - 1];
    } else {
      return -1;
    }
  }

  public int peek(int stackNum) {
    int idx = locations[stackNum];
    if (idx > stackNum * size) {
      return data[idx - 1];
    } else {
      return -1;
    }
  }

  public boolean isEmpty(int stackNum) {
    return locations[stackNum] == stackNum * size;
  }
}
