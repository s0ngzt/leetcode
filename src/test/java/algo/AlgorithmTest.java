package algo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class AlgorithmTest {

  BinarySearch algo = new BinarySearch();

  @Test
  void testBinarySearch() {
    int[] intArray = {1, 2, 2, 2, 3, 4, 6};
    assertEquals(-1, algo.binarySearch(intArray, 0));
    assertEquals(3, algo.binarySearch(intArray, 2));
    assertEquals(5, algo.binarySearch(intArray, 4));
    assertEquals(-1, algo.binarySearch(intArray, 8));
  }

  @Test
  void testLeftAndRightBound() {
    int[] intArray = {1, 2, 2, 2, 3, 4, 6};
    assertEquals(1, algo.leftBound(intArray, 2));
    assertEquals(3, algo.rightBound(intArray, 2));
    assertEquals(5, algo.leftBound(intArray, 4));
    assertEquals(5, algo.rightBound(intArray, 4));

    assertEquals(0, algo.rightBound(intArray, 1));
  }
}
