package leetcode;

class MyHashSet {

    boolean[] data;

    public MyHashSet() {
        data = new boolean[1000000];
    }

    public void add(int key) {
        data[key] = true;
    }

    public void remove(int key) {
        data[key] = false;
    }

    public boolean contains(int key) {
        return data[key];
    }

}
