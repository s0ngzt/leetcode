package interview;

import java.util.LinkedList;
import java.util.Queue;

// 面试题 03.06. 动物收容所
// FIFO
class AnimalShelf {

    Queue<int[]> cats;
    Queue<int[]> dogs;

    public AnimalShelf() {
        cats = new LinkedList<>();
        dogs = new LinkedList<>();
    }

    public void enqueue(int[] animal) {
        // 0 cat / 1 dog
        if (animal[1] == 0) {
            cats.offer(animal);
        } else if (animal[1] == 1) {
            dogs.offer(animal);
        }
    }

    public int[] dequeueAny() {
        if (cats.isEmpty() && dogs.isEmpty()) {
            return new int[]{-1, -1};
        } else if (cats.isEmpty()) {
            return dogs.poll();
        } else if (dogs.isEmpty()) {
            return cats.poll();
        } else {
            var oldestCat = cats.peek();
            var oldestDog = dogs.peek();
            if (oldestCat[0] <= oldestDog[0]) {
                return cats.poll();
            } else {
                return dogs.poll();
            }
        }
    }

    public int[] dequeueDog() {
        if (!dogs.isEmpty()) {
            return dogs.poll();
        } else {
            return new int[]{-1, -1};
        }
    }

    public int[] dequeueCat() {
        if (!cats.isEmpty()) {
            return cats.poll();
        } else {
            return new int[]{-1, -1};
        }
    }
}
