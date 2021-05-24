package offer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class SolutionTest {

  Solution solution = new Solution();

  @Test
  void testShipWithinDays() {
    assertEquals("world hello,", solution.reverseWords("hello, world"));
  }


}
