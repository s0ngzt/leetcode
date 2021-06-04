package codewars;

import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.Collectors;

class Solution {

  // 6 kyu
  public static String high(String s) {
    Function<String, Integer> score = (String word) -> {
      int sum = 0;
      for (char c : word.toCharArray()) {
        sum += c - 96;
      }
      return sum;
    };
    var sorted = Arrays.stream(s.split(" ")).sorted(
        (o1, o2) -> score.apply(o2) - score.apply(o1));

    return sorted.collect(Collectors.toList()).get(0);
  }
}
