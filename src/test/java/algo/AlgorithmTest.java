package algo;

import org.junit.jupiter.api.Test;

public class AlgorithmTest {


  @Test
  void test() {
    BinarySearchTree<Integer, Integer> bst = new BinarySearchTree<>();
    bst.insert(61, 2);
    bst.insert(40, 2);
    bst.insert(98, 2);
    bst.print();
    bst.insert(32, 2);
    bst.insert(71, 2);
    bst.insert(72, 2);
    bst.print();
    bst.insert(1, 2);

    bst.delete(40);
    bst.print();
    bst.delete(98);
    bst.print();

  }

}
