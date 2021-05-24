package leetcode;


import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class SolutionMediumTest {

  SolutionMedium solution = new SolutionMedium();

  @Test
  void testShipWithinDays() {
    assertEquals(15, solution.shipWithinDays(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5));
    assertEquals(9, solution.shipWithinDays(new int[]{1, 2, 3, 4, 5}, 2));
  }

  @Test
  void testDecodeString() {
    assertEquals("aaabcbc", solution.decodeString("3[a]2[bc]"));
    assertEquals("accaccacc", solution.decodeString("3[a2[c]]"));
    assertEquals("abcabccdcdcdef", solution.decodeString("2[abc]3[cd]ef"));
    assertEquals("abccdcdcdxyz", solution.decodeString("abc3[cd]xyz"));
  }

}
