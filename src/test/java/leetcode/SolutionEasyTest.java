package leetcode;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class SolutionEasyTest {

  SolutionEasy solution = new SolutionEasy();

  @Test
  void testHamming() {
    assertEquals(2, solution.hammingDistance(1, 4));
  }


}
